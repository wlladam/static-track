"""Moves an MP4's `moov` atom (the index OpenCV's writer appends at the end
of the file) to the front, right after `ftyp` - the "faststart" layout.

WHY THIS EXISTS
overlay_debug.py's DebugVideoWriter uses OpenCV's built-in MP4 muxer, which
writes `ftyp`, then `mdat` (the actual frame data, streamed out as frames
arrive), then `moov` (the index) last, once the total frame count is known.
Desktop browsers tolerate this - they buffer/seek within the downloaded
file freely. iOS Safari's <video> element does not: it needs to read
`moov` before it can begin decoding anything, and won't fetch it from the
end of a multi-megabyte file on its own, so the video silently fails to
ever start playing - the exact "doesn't show up on iPhone" symptom, while
the same file plays fine everywhere else. This is normally fixed by
`ffmpeg -movflags faststart`, but there's no standalone ffmpeg binary
available here (OpenCV links FFmpeg as a library, not a CLI tool) - this
reimplements the same relocation ffmpeg/qt-faststart do, in pure Python,
so no new system dependency is needed in production.

HOW IT WORKS
1. Parse the file's top-level boxes (ftyp/free/mdat/moov/...).
2. If `moov` already comes before `mdat`, the file is already faststart -
   nothing to do (also makes this safe to call unconditionally/idempotently).
3. Cut `moov`'s bytes out and reinsert them immediately after `ftyp`.
4. Every sample's absolute byte offset is recorded inside `moov` (in each
   track's `stco`/`co64` boxes) relative to the file's start - moving
   `moov` earlier shifts every byte after the insertion point later by
   len(moov), so those offsets must be corrected by the same amount or the
   file is a valid-looking MP4 that points at the wrong bytes for every
   sample.
"""
import shutil
import struct
import tempfile
from pathlib import Path

_TOP_LEVEL_LEAF_BOXES = {b"stco", b"co64"}
_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}


def _read_boxes(data: bytes, start: int, end: int) -> list:
    """[(type: bytes, offset: int, size: int, header_len: int), ...] for
    every box in data[start:end]. 64-bit ("largesize") boxes are supported
    since a multi-gigabyte mdat (unlikely here, but real HD footage could
    get close) would otherwise be misparsed.
    """
    boxes = []
    pos = start
    while pos + 8 <= end:
        size32 = struct.unpack(">I", data[pos : pos + 4])[0]
        box_type = bytes(data[pos + 4 : pos + 8])
        header_len = 8
        size = size32
        if size32 == 1:
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header_len = 16
        elif size32 == 0:
            size = end - pos  # box extends to the end of its parent
        if size < header_len:
            break  # malformed - bail out rather than looping forever
        boxes.append((box_type, pos, size, header_len))
        pos += size
    return boxes


def _shift_chunk_offsets(moov: bytearray, shift: int) -> None:
    """Walks every stco/co64 box inside `moov` (recursively, through
    trak/mdia/minf/stbl) and adds `shift` to every recorded chunk offset -
    see module docstring for why this has to happen when moov moves.
    """
    def walk(data: bytearray, start: int, end: int):
        for box_type, offset, size, header_len in _read_boxes(data, start, end):
            body_start = offset + header_len
            if box_type == b"stco":
                # version/flags (4 bytes) + entry_count (4 bytes), then
                # entry_count * 4-byte (32-bit) chunk offsets.
                entry_count = struct.unpack(">I", data[body_start + 4 : body_start + 8])[0]
                p = body_start + 8
                for _ in range(entry_count):
                    val = struct.unpack(">I", data[p : p + 4])[0]
                    struct.pack_into(">I", data, p, val + shift)
                    p += 4
            elif box_type == b"co64":
                entry_count = struct.unpack(">I", data[body_start + 4 : body_start + 8])[0]
                p = body_start + 8
                for _ in range(entry_count):
                    val = struct.unpack(">Q", data[p : p + 8])[0]
                    struct.pack_into(">Q", data, p, val + shift)
                    p += 8
            elif box_type in _CONTAINER_BOXES:
                walk(data, body_start, offset + size)

    walk(moov, 0, len(moov))


def make_faststart(path: str) -> bool:
    """Rewrites the MP4 at `path` in place so `moov` comes before `mdat`.
    Returns True if a rewrite happened, False if the file was already
    faststart (or isn't a box-structured MP4 this parser recognizes - in
    which case it's left untouched rather than risking corrupting it).
    """
    data = Path(path).read_bytes()
    boxes = _read_boxes(data, 0, len(data))
    box_types = [b[0] for b in boxes]
    if b"moov" not in box_types or b"mdat" not in box_types:
        return False
    if box_types.index(b"moov") < box_types.index(b"mdat"):
        return False  # already faststart

    moov_type, moov_offset, moov_size, _ = next(b for b in boxes if b[0] == b"moov")
    moov_bytes = bytearray(data[moov_offset : moov_offset + moov_size])

    # Insertion point: right after the last box that precedes mdat and
    # isn't moov itself (typically just `ftyp`) - i.e. the start of mdat,
    # since moov is being pulled out from wherever it currently sits.
    mdat_offset = next(b[1] for b in boxes if b[0] == b"mdat")
    insert_at = mdat_offset

    # Every chunk offset inside moov is measured from the start of the
    # file. Moving moov to `insert_at` pushes everything currently at or
    # after `insert_at` later by len(moov) - correct the offsets to match.
    _shift_chunk_offsets(moov_bytes, moov_size)

    before = data[:insert_at]
    after_without_moov = data[insert_at:moov_offset] + data[moov_offset + moov_size :]

    with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as tmp:
        tmp.write(before)
        tmp.write(moov_bytes)
        tmp.write(after_without_moov)
        tmp_path = tmp.name

    shutil.move(tmp_path, path)
    return True
