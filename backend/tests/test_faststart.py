"""Unit tests for pipeline/faststart.py, using a synthetic minimal MP4 box
structure (not a real video) so the box-relocation logic itself is verified
independent of OpenCV/real encoding - mirrors this project's existing
convention (test_hold_detection.py, test_variant_classification.py) of
testing pure logic against constructed fixtures.
"""
import struct

from pipeline.faststart import make_faststart


def _box(box_type: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + box_type + body


def _stco(offsets: list) -> bytes:
    body = struct.pack(">II", 0, len(offsets)) + b"".join(struct.pack(">I", o) for o in offsets)
    return _box(b"stco", body)


def _co64(offsets: list) -> bytes:
    body = struct.pack(">II", 0, len(offsets)) + b"".join(struct.pack(">Q", o) for o in offsets)
    return _box(b"co64", body)


def _wrap(box_type: bytes, *children: bytes) -> bytes:
    return _box(box_type, b"".join(children))


def _build_mp4(mdat_payload: bytes, offsets_into_mdat: list, use_co64: bool = False) -> bytes:
    """ftyp, mdat (containing `mdat_payload` at the given byte offsets
    *within the payload*), then moov (with a real, nested stco/co64
    pointing at the corresponding *absolute file* offsets) - the layout
    OpenCV's muxer actually produces (moov last).
    """
    ftyp = _box(b"ftyp", b"isomiso2avc1mp41")
    mdat_offset = len(ftyp)
    mdat = _box(b"mdat", mdat_payload)
    mdat_body_start = mdat_offset + 8

    absolute_offsets = [mdat_body_start + o for o in offsets_into_mdat]
    stco_or_co64 = _co64(absolute_offsets) if use_co64 else _stco(absolute_offsets)
    stbl = _wrap(b"stbl", stco_or_co64)
    minf = _wrap(b"minf", stbl)
    mdia = _wrap(b"mdia", minf)
    trak = _wrap(b"trak", mdia)
    moov = _wrap(b"moov", trak)

    return ftyp + mdat + moov, absolute_offsets, mdat_body_start


def _parse_top_level(data: bytes) -> list:
    pos = 0
    boxes = []
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        boxes.append((data[pos + 4 : pos + 8], pos, size))
        if size == 0:
            break
        pos += size
    return boxes


def _read_stco_offsets(data: bytes) -> list:
    idx = data.index(b"stco")
    body_start = idx + 4  # past the type, into the body
    entry_count = struct.unpack(">I", data[body_start + 4 : body_start + 8])[0]
    p = body_start + 8
    return [struct.unpack(">I", data[p + 4 * i : p + 4 * i + 4])[0] for i in range(entry_count)]


def _read_co64_offsets(data: bytes) -> list:
    idx = data.index(b"co64")
    body_start = idx + 4
    entry_count = struct.unpack(">I", data[body_start + 4 : body_start + 8])[0]
    p = body_start + 8
    return [struct.unpack(">Q", data[p + 8 * i : p + 8 * i + 8])[0] for i in range(entry_count)]


def test_moov_moves_before_mdat(tmp_path):
    payload = b"A" * 5 + b"B" * 5 + b"C" * 5
    data, _, _ = _build_mp4(payload, offsets_into_mdat=[0, 5, 10])
    path = tmp_path / "clip.mp4"
    path.write_bytes(data)

    changed = make_faststart(str(path))

    assert changed is True
    boxes = _parse_top_level(path.read_bytes())
    box_order = [b[0] for b in boxes]
    assert box_order.index(b"moov") < box_order.index(b"mdat")
    assert box_order[0] == b"ftyp"


def test_chunk_offsets_are_corrected_after_the_move(tmp_path):
    # Real-world-shaped case: several samples at distinct byte offsets
    # inside mdat - every one of them must still point at the exact same
    # bytes after moov relocates.
    payload = b"".join(bytes([i]) * 100 for i in range(5))  # 5 x 100-byte "samples"
    sample_offsets = [i * 100 for i in range(5)]
    data, expected_absolute_offsets, mdat_body_start = _build_mp4(payload, sample_offsets)
    path = tmp_path / "clip.mp4"
    path.write_bytes(data)

    make_faststart(str(path))
    result = path.read_bytes()

    new_offsets = _read_stco_offsets(result)
    for new_offset, sample_index in zip(new_offsets, range(5)):
        # The byte at the corrected offset must be the same "sample" byte
        # value it always was - proof the offset still points at the right
        # data, not just a plausible-looking number.
        assert result[new_offset] == sample_index


def test_co64_offsets_are_also_corrected(tmp_path):
    payload = b"X" * 10 + b"Y" * 10
    data, _, _ = _build_mp4(payload, offsets_into_mdat=[0, 10], use_co64=True)
    path = tmp_path / "clip.mp4"
    path.write_bytes(data)

    make_faststart(str(path))
    result = path.read_bytes()

    offsets = _read_co64_offsets(result)
    assert result[offsets[0] : offsets[0] + 10] == b"X" * 10
    assert result[offsets[1] : offsets[1] + 10] == b"Y" * 10


def test_already_faststart_file_is_left_unchanged(tmp_path):
    payload = b"Z" * 20
    data, _, _ = _build_mp4(payload, offsets_into_mdat=[0])
    # Manually reorder to already be faststart (moov before mdat).
    boxes = _parse_top_level(data)
    ftyp_bytes = data[boxes[0][1] : boxes[0][1] + boxes[0][2]]
    mdat_bytes = data[boxes[1][1] : boxes[1][1] + boxes[1][2]]
    moov_bytes = data[boxes[2][1] : boxes[2][1] + boxes[2][2]]
    already_faststart = ftyp_bytes + moov_bytes + mdat_bytes

    path = tmp_path / "clip.mp4"
    path.write_bytes(already_faststart)

    changed = make_faststart(str(path))

    assert changed is False
    assert path.read_bytes() == already_faststart


def test_file_size_is_unchanged_by_the_relocation(tmp_path):
    payload = b"Q" * 200
    data, _, _ = _build_mp4(payload, offsets_into_mdat=[0, 50, 100, 150])
    path = tmp_path / "clip.mp4"
    path.write_bytes(data)
    original_size = len(data)

    make_faststart(str(path))

    assert path.stat().st_size == original_size


def test_non_mp4_file_is_left_untouched(tmp_path):
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"this is not a real mp4 file at all")

    changed = make_faststart(str(path))

    assert changed is False
