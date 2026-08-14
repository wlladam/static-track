"""Integration tests for the Flask routes.

process_video is monkeypatched for most tests so they run fast and don't
depend on real MediaPipe processing - the pipeline itself already has
thorough unit test coverage (test_hold_detection.py, test_scoring.py, etc.)
and was verified against real videos manually.
"""
import io

from app import routes as routes_module
from app.models import Attempt


def _fake_video_bytes():
    return io.BytesIO(b"not a real video, just bytes for file.save()")


def test_index_with_empty_db(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"ANALYZE" in resp.data and b"HOLD" in resp.data


def test_history_with_empty_db(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"History" in resp.data


def test_upload_with_no_file_flashes_and_redirects(client):
    resp = client.post("/upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert resp.location == "/"

    follow = client.get("/")
    assert b"Please choose a video file" in follow.data


def test_upload_with_bad_extension_flashes_and_redirects(client):
    resp = client.post(
        "/upload",
        data={"video": (io.BytesIO(b"hello"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302

    follow = client.get("/")
    assert b"Unsupported file type" in follow.data


def test_upload_without_movement_type_flashes_and_redirects(client):
    resp = client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    assert resp.location == "/"

    follow = client.get("/")
    assert b"select whether this is a static hold, a dynamic rep set, or a combo" in follow.data


def test_upload_success_creates_attempt_and_renders_report(client, app, monkeypatch):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 3.0,
        "move_type": "front_lever",
        "progression": "full",
        "overall_score": 90.0,
        "overall_confidence": "high",
        "report_json": (
            '{"features": {}, "criteria": {"arm_lockout": {"score": 90.0, '
            '"label": "excellent lockout", "confidence": "high", "detail": {}}}, '
            '"strengths": [], "refine": [], "weaknesses": [], "summary": "s", "scapular_position_note": "note"}'
        ),
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)

    resp = client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    assert resp.location.startswith("/attempts/")

    with app.app_context():
        attempt = Attempt.query.first()
        assert attempt is not None
        assert attempt.original_filename == "clip.mp4"
        assert attempt.overall_score == 90.0

    report_resp = client.get(resp.location)
    assert report_resp.status_code == 200
    assert b"front lever" in report_resp.data
    assert b"90.0" in report_resp.data


def test_upload_pipeline_exception_stores_error_not_500(client, app, monkeypatch):
    def _boom(video_path, data_dir=None, movement_type_hint=None, progression_hint=None):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(routes_module, "process_video", _boom)

    resp = client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302

    report_resp = client.get(resp.location)
    assert report_resp.status_code == 200
    assert b"simulated pipeline failure" in report_resp.data

    with app.app_context():
        attempt = Attempt.query.first()
        assert attempt.error == "simulated pipeline failure"
        assert attempt.hold_detected is False


def test_report_404_for_missing_attempt(client):
    resp = client.get("/attempts/999")
    assert resp.status_code == 404


def test_overlay_video_404_when_no_overlay(client, monkeypatch):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": False,
        "start_sec": None,
        "end_sec": None,
        "duration_sec": None,
        "move_type": None,
        "progression": None,
        "overall_score": None,
        "overall_confidence": None,
        "report_json": None,
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)

    resp = client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
        content_type="multipart/form-data",
    )
    attempt_url = resp.location

    overlay_resp = client.get(attempt_url.replace("/attempts/", "/media/overlay/"))
    assert overlay_resp.status_code == 404


def test_history_lists_attempts_and_renders_chart_with_two_scores(client, monkeypatch):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 3.0,
        "move_type": "front_lever",
        "progression": "full",
        "overall_score": 80.0,
        "overall_confidence": "high",
        "report_json": '{"features": {}, "criteria": {}, "strengths": [], "refine": [], "weaknesses": [], "summary": "s", "scapular_position_note": "note"}',
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)

    for _ in range(2):
        client.post(
            "/upload",
            data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
            content_type="multipart/form-data",
        )

    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"80.0/100" in resp.data  # table row
    assert b"<svg" in resp.data  # chart renders once there are 2+ scored attempts
    assert b"Sessions logged" in resp.data
    assert b"Best Difficulty Scaler" in resp.data


def test_history_summary_stats_reflect_logged_sessions(client, monkeypatch):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 4.0,
        "move_type": "front_lever",
        "progression": "full",
        "overall_score": 92.0,
        "overall_confidence": "high",
        "report_json": "{}",
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)
    client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
        content_type="multipart/form-data",
    )

    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"92.0" in resp.data


def test_history_range_filter_hides_older_sessions(client, monkeypatch, app):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 4.0,
        "move_type": "front_lever",
        "progression": "full",
        "overall_score": 77.0,
        "overall_confidence": "high",
        "report_json": "{}",
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)
    client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
        content_type="multipart/form-data",
    )

    from datetime import datetime, timedelta, timezone

    from app.models import Attempt, db

    with app.app_context():
        attempt = Attempt.query.first()
        attempt.uploaded_at = datetime.now(timezone.utc) - timedelta(days=60)
        db.session.commit()

    resp = client.get("/history?range=7")
    assert resp.status_code == 200
    assert b"No sessions match the current filters" in resp.data

    resp_all = client.get("/history?range=all")
    assert b"77.0" in resp_all.data


def test_history_movement_drilldown_shows_prs(client, monkeypatch):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "movement_type": "static_hold",
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 3.0,
        "move_type": "front_lever",
        "progression": "full",
        "overall_score": 70.0,
        "overall_confidence": "high",
        "report_json": "{}",
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)
    for score in (70.0, 95.0):
        fake_result["overall_score"] = score
        client.post(
            "/upload",
            data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
            content_type="multipart/form-data",
        )

    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"Best Difficulty Scaler" in resp.data
    assert b"142.5" in resp.data  # 95.0 raw * 1.5x full front lever multiplier


def test_history_table_sort_by_score_ascending(client, monkeypatch):
    fake_result = {
        "debug_overlay_path": None,
        "hold_detected": True,
        "start_sec": 1.0,
        "end_sec": 4.0,
        "duration_sec": 3.0,
        "move_type": "front_lever",
        "progression": "full",
        "overall_score": 70.0,
        "overall_confidence": "high",
        "report_json": "{}",
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)
    for score in (70.0, 95.0):
        fake_result["overall_score"] = score
        client.post(
            "/upload",
            data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
            content_type="multipart/form-data",
        )

    resp = client.get("/history?sort=score&dir=asc&group=none")
    assert resp.status_code == 200
    assert resp.data.index(b"70.0/100") < resp.data.index(b"95.0/100")


def test_upload_dynamic_reps_renders_report(client, monkeypatch):
    fake_result = {
        "hold_detected": True,
        "movement_type": "dynamic_reps",
        "start_sec": None,
        "end_sec": None,
        "duration_sec": None,
        "move_type": "front_lever",
        "progression": None,
        "overall_score": 88.5,
        "overall_confidence": "high",
        "report_json": (
            '{"strengths": [{"criterion": "arm_lockout", "label": "Arm lockout", "kind": "strength", '
            '"headline": "Elbows stayed locked out at 178.0deg.", "context": "Great extension.", '
            '"severity": 10.0, "score": 95.0, "direction": null}], '
            '"refine": [], "weaknesses": [], "recommendations": [], '
            '"summary": "Strong front lever pull up set overall (88.5/100)."}'
        ),
        "exercise_type": "front_lever_pull_up",
        "rep_count": 3,
        "avg_rep_duration_sec": 2.1,
        "rom_consistency_score": 92.0,
        "reps_json": (
            '[{"index": 1, "start_sec": 0.2, "peak_sec": 1.0, "end_sec": 2.3, '
            '"duration_sec": 2.1, "rom": 0.18, "move_type": "front_lever", '
            '"progression": "tuck", "arm_lockout_score": 82.0, "hip_shoulder_score": 75.0}]'
        ),
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)

    resp = client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "static_hold"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302

    report_resp = client.get(resp.location)
    assert report_resp.status_code == 200
    assert b"front lever pull up" in report_resp.data
    assert b"3" in report_resp.data  # rep count
    assert b"88.5" in report_resp.data  # form score, now the headline stat instead of ROM consistency
    assert b"Strong front lever pull up set overall" in report_resp.data  # summary
    assert b"Elbows stayed locked out" in report_resp.data  # strengths section headline


def test_upload_combo_renders_report(client, monkeypatch):
    fake_result = {
        "hold_detected": True,
        "movement_type": "combo",
        "start_sec": None,
        "end_sec": None,
        "duration_sec": None,
        "move_type": None,
        "progression": None,
        "overall_score": None,
        "overall_confidence": None,
        "report_json": '{"summary": "2-move combo: tuck front lever -> full front lever."}',
        "exercise_type": None,
        "rep_count": None,
        "avg_rep_duration_sec": None,
        "rom_consistency_score": None,
        "reps_json": (
            '[{"index": 1, "move_type": "front_lever", "progression": "tuck", "kind": "hold", '
            '"start_sec": 0.2, "end_sec": 2.2, "duration_sec": 2.0, "score": 80.0, '
            '"critique": "Arm lockout: Elbows stayed locked out."}, '
            '{"index": 2, "move_type": "front_lever", "progression": "full", "kind": "touch", '
            '"start_sec": 3.0, "end_sec": 3.4, "duration_sec": 0.4, "score": 60.0, '
            '"critique": "Hip/shoulder alignment: Hips sagged slightly below the line."}]'
        ),
    }
    monkeypatch.setattr(routes_module, "process_video", lambda video_path, data_dir=None, movement_type_hint=None, progression_hint=None: fake_result)

    resp = client.post(
        "/upload",
        data={"video": (_fake_video_bytes(), "clip.mp4"), "movement_type": "combo"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302

    report_resp = client.get(resp.location)
    assert report_resp.status_code == 200
    assert b"2-move combo" in report_resp.data
    assert b"tuck" in report_resp.data and b"full" in report_resp.data
    assert b"touch" in report_resp.data  # kind badge
    assert b"Elbows stayed locked out" in report_resp.data
    assert b"Hips sagged slightly below the line" in report_resp.data
