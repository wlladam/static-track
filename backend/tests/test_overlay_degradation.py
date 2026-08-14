"""Regression tests for the "analysis crashes with [Errno 2] No such file
or directory" bug: on some hosts (confirmed on Render's Linux container)
cv2.VideoWriter's H.264 encoder can fail to actually open/produce a file
even when instantiating it doesn't raise - the overlay is a nice-to-have,
so this must degrade gracefully instead of crashing the whole analysis.
"""
import numpy as np
import pytest

from pipeline.overlay_debug import DebugVideoWriter


class _FakeCvWriter:
    """Stands in for cv2.VideoWriter without needing a real encoder -
    lets tests deterministically simulate "opened successfully" vs
    "silently failed to open" without depending on what codecs happen to
    be available on whatever machine runs the test suite.
    """

    def __init__(self, opened: bool):
        self._opened = opened
        self.write_calls = 0

    def isOpened(self):
        return self._opened

    def write(self, frame):
        self.write_calls += 1

    def release(self):
        pass


def test_is_open_true_when_encoder_opens_successfully(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.overlay_debug.cv2.VideoWriter", lambda *a, **k: _FakeCvWriter(opened=True))

    writer = DebugVideoWriter(str(tmp_path / "clip.mp4"), fps=5.0, frame_size=(100, 100))

    assert writer.is_open is True


def test_is_open_false_when_encoder_fails_to_open(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.overlay_debug.cv2.VideoWriter", lambda *a, **k: _FakeCvWriter(opened=False))

    writer = DebugVideoWriter(str(tmp_path / "clip.mp4"), fps=5.0, frame_size=(100, 100))

    assert writer.is_open is False


def test_write_is_a_silent_no_op_when_encoder_never_opened(monkeypatch, tmp_path):
    fake = _FakeCvWriter(opened=False)
    monkeypatch.setattr("pipeline.overlay_debug.cv2.VideoWriter", lambda *a, **k: fake)

    writer = DebugVideoWriter(str(tmp_path / "clip.mp4"), fps=5.0, frame_size=(100, 100))
    # Must not raise, even though there's nothing real underneath.
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))

    assert fake.write_calls == 0


def test_write_still_forwards_to_a_real_working_encoder(monkeypatch, tmp_path):
    fake = _FakeCvWriter(opened=True)
    monkeypatch.setattr("pipeline.overlay_debug.cv2.VideoWriter", lambda *a, **k: fake)

    writer = DebugVideoWriter(str(tmp_path / "clip.mp4"), fps=5.0, frame_size=(100, 100))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))

    assert fake.write_calls == 1
