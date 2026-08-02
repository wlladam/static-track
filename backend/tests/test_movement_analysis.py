"""Unit tests for movement_analysis.py's dynamic-rep orchestration, using
synthetic landmark sets (mirrors the approach in test_variant_classification.py
and test_rep_detection.py - no real planche/front-lever pull-up footage with
2+ reps exists yet).
"""
import copy

from pipeline.movement_analysis import analyze_movement

FPS = 5.0

# A straight-body front lever: legs straight and together, arms straight,
# wrists gripping above the shoulder line (same as test_variant_classification.py).
FRONT_LEVER_LANDMARKS = {
    "left_shoulder": {"x": 0.40, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "right_shoulder": {"x": 0.40, "y": 0.35, "z": 0.0, "visibility": 1.0},
    "left_elbow": {"x": 0.30, "y": 0.25, "z": 0.0, "visibility": 1.0},
    "right_elbow": {"x": 0.30, "y": 0.30, "z": 0.0, "visibility": 1.0},
    "left_wrist": {"x": 0.20, "y": 0.20, "z": 0.0, "visibility": 1.0},
    "right_wrist": {"x": 0.20, "y": 0.25, "z": 0.0, "visibility": 1.0},
    "left_hip": {"x": 0.55, "y": 0.50, "z": 0.0, "visibility": 1.0},
    "right_hip": {"x": 0.55, "y": 0.55, "z": 0.0, "visibility": 1.0},
    "left_knee": {"x": 0.70, "y": 0.50, "z": 0.0, "visibility": 1.0},
    "right_knee": {"x": 0.70, "y": 0.55, "z": 0.0, "visibility": 1.0},
    "left_ankle": {"x": 0.85, "y": 0.50, "z": 0.0, "visibility": 1.0},
    "right_ankle": {"x": 0.85, "y": 0.55, "z": 0.0, "visibility": 1.0},
}

# A "touch" front lever: hips pulled in close to the wrists (the bar/anchor)
# - a touch's defining feature, per classify_variant's TOUCH_GAP_RATIO_THRESHOLD
# docstring, is hip-to-bar contact, not how briefly it's held.
TOUCH_FRONT_LEVER_LANDMARKS = {
    **copy.deepcopy(FRONT_LEVER_LANDMARKS),
    "left_hip": {"x": 0.25, "y": 0.22, "z": 0.0, "visibility": 1.0},
    "right_hip": {"x": 0.25, "y": 0.27, "z": 0.0, "visibility": 1.0},
    "left_knee": {"x": 0.40, "y": 0.22, "z": 0.0, "visibility": 1.0},
    "right_knee": {"x": 0.40, "y": 0.27, "z": 0.0, "visibility": 1.0},
    "left_ankle": {"x": 0.55, "y": 0.22, "z": 0.0, "visibility": 1.0},
    "right_ankle": {"x": 0.55, "y": 0.27, "z": 0.0, "visibility": 1.0},
}

# Wrists below the hip line -> planche, per classify_variant's move_type rule.
PLANCHE_LANDMARKS = {
    **copy.deepcopy(FRONT_LEVER_LANDMARKS),
    "left_wrist": {"x": 0.20, "y": 0.60, "z": 0.0, "visibility": 1.0},
    "right_wrist": {"x": 0.20, "y": 0.65, "z": 0.0, "visibility": 1.0},
    "left_elbow": {"x": 0.30, "y": 0.45, "z": 0.0, "visibility": 1.0},
    "right_elbow": {"x": 0.30, "y": 0.50, "z": 0.0, "visibility": 1.0},
}


def _pose(base, **overrides):
    landmarks = copy.deepcopy(base)
    for joint, values in overrides.items():
        landmarks[joint].update(values)
    return landmarks


# Bent knees (tuck) - same shape as test_variant_classification.py's tuck case.
TUCK_LANDMARKS = _pose(
    FRONT_LEVER_LANDMARKS,
    left_knee={"x": 0.65, "y": 0.55},
    left_ankle={"x": 0.60, "y": 0.65},
    right_knee={"x": 0.65, "y": 0.60},
    right_ankle={"x": 0.60, "y": 0.70},
)


def _shift_all(landmarks, dx, dy):
    return {joint: {**v, "x": v["x"] + dx, "y": v["y"] + dy} for joint, v in landmarks.items()}


def _combo_records(move_landmarks_list, cluster_len=8, transition_len=3, fps=FPS):
    """Builds a combo clip: for each (landmarks, is_touch) in
    move_landmarks_list, a stable cluster at that position (short if
    is_touch), separated by large-displacement "jump" transitions.
    """
    frames = []
    for move_i, (landmarks, is_touch) in enumerate(move_landmarks_list):
        # 4 frames for a touch: the first still carries transition
        # displacement from the preceding jump (see test_hold_detection.py),
        # so only the last 3 (0.4s at 5fps) read as truly stable - above
        # detect_all_holds' 0.3s touch floor but well under a sustained hold.
        n = 4 if is_touch else cluster_len
        for i in range(n):
            frames.append(_shift_all(landmarks, 0.001 * ((-1) ** i), 0.0))
        if move_i < len(move_landmarks_list) - 1:
            for i in range(transition_len):
                frames.append(_shift_all(landmarks, 0.3 * ((-1) ** i), 0.2))
    return [
        {"frame_index": i, "timestamp_sec": round(i / fps, 3), "landmarks": lm} for i, lm in enumerate(frames)
    ]


def _triangle_wave_records(base_landmarks, n_reps, bottom=0.50, top=0.25, steps_per_half=5):
    """A repeating bottom->top->bottom hip-height oscillation, with straight
    elbows at the trough (bottom, i.e. arms locked) bending sharply at each
    peak (top) to give real elbow ROM - mimics a pull-up/push-up's arm-driven
    motion, not a body-driven raise.
    """
    hip_ys = [bottom]
    for _ in range(n_reps):
        for i in range(1, steps_per_half + 1):
            hip_ys.append(bottom + (top - bottom) * i / steps_per_half)
        for i in range(1, steps_per_half + 1):
            hip_ys.append(top + (bottom - top) * i / steps_per_half)

    base_hip_y = base_landmarks["left_hip"]["y"]
    records = []
    for i, hip_y in enumerate(hip_ys):
        is_peak = abs(hip_y - top) < 1e-6
        bend = {"left_elbow": {"y": 0.05}, "right_elbow": {"y": 0.10}} if is_peak else {}
        # Shift the whole body (hip/knee/ankle) by the same delta so the leg
        # stays straight (hip-knee-ankle collinear) throughout - only the
        # elbows bend at the peak, isolating "arm ROM" as the rep signal.
        dy = hip_y - base_hip_y
        landmarks = _pose(
            base_landmarks,
            left_hip={"y": hip_y},
            right_hip={"y": hip_y + 0.05},
            left_knee={"y": base_landmarks["left_knee"]["y"] + dy},
            right_knee={"y": base_landmarks["right_knee"]["y"] + dy},
            left_ankle={"y": base_landmarks["left_ankle"]["y"] + dy},
            right_ankle={"y": base_landmarks["right_ankle"]["y"] + dy},
            **bend,
        )
        records.append({"frame_index": i, "timestamp_sec": round(i / FPS, 3), "landmarks": landmarks})
    return records


def test_planche_rep_names_exercise_push_up_not_pull_up():
    # "planche_pull_up" is nonsensical - a planche is pressed into, never pulled.
    records = _triangle_wave_records(PLANCHE_LANDMARKS, n_reps=2)

    outcome = analyze_movement(records, movement_type_hint="dynamic_reps")

    assert outcome is not None
    kind, dynamic = outcome
    assert kind == "dynamic_reps"
    assert dynamic.move_type == "planche"
    assert dynamic.exercise_type == "planche_push_up"


def test_front_lever_rep_names_exercise_pull_up():
    records = _triangle_wave_records(FRONT_LEVER_LANDMARKS, n_reps=2)

    outcome = analyze_movement(records, movement_type_hint="dynamic_reps")

    assert outcome is not None
    _, dynamic = outcome
    assert dynamic.exercise_type == "front_lever_pull_up"


def test_single_rep_rom_consistency_is_none_not_a_fabricated_100():
    # ROM consistency is a measure of variation *across* reps - with only one
    # rep there's nothing to compare, so it must not report a number.
    records = _triangle_wave_records(FRONT_LEVER_LANDMARKS, n_reps=1)

    outcome = analyze_movement(records, movement_type_hint="dynamic_reps")

    assert outcome is not None
    _, dynamic = outcome
    assert dynamic.rep_count == 1
    assert dynamic.rom_consistency_score is None


def test_multi_rep_rom_consistency_is_a_real_number():
    records = _triangle_wave_records(FRONT_LEVER_LANDMARKS, n_reps=3)

    outcome = analyze_movement(records, movement_type_hint="dynamic_reps")

    assert outcome is not None
    _, dynamic = outcome
    assert dynamic.rep_count == 3
    assert dynamic.rom_consistency_score is not None


def test_progression_hint_overrides_straddle_vs_full_for_reps():
    records = _triangle_wave_records(FRONT_LEVER_LANDMARKS, n_reps=1)

    outcome = analyze_movement(records, movement_type_hint="dynamic_reps", progression_hint="straddle")

    assert outcome is not None
    _, dynamic = outcome
    assert dynamic.reps[0].progression == "straddle"


def test_combo_finds_each_move_in_chronological_order():
    records = _combo_records([(TUCK_LANDMARKS, False), (FRONT_LEVER_LANDMARKS, False)])

    outcome = analyze_movement(records, movement_type_hint="combo")

    assert outcome is not None
    kind, combo = outcome
    assert kind == "combo"
    assert len(combo.moves) == 2
    assert combo.moves[0].progression == "tuck"
    assert combo.moves[0].kind == "hold"
    assert combo.moves[1].progression == "full"
    assert combo.moves[1].kind == "hold"
    assert combo.moves[0].start_sec < combo.moves[1].start_sec


def test_combo_detects_a_touch_front_lever_by_hip_bar_contact_not_duration():
    # A real touch front lever was held a full ~2 seconds - the same
    # duration as the non-touching full front lever right after it in the
    # same clip - so duration can't be what distinguishes them. Hip-to-wrist
    # (bar) proximity is the real signal.
    records = _combo_records([(TOUCH_FRONT_LEVER_LANDMARKS, False), (FRONT_LEVER_LANDMARKS, False)])

    outcome = analyze_movement(records, movement_type_hint="combo")

    assert outcome is not None
    _, combo = outcome
    assert len(combo.moves) == 2
    assert combo.moves[0].kind == "touch"
    assert combo.moves[1].kind == "hold"
    # Both moves held for the same (non-brief) duration - proves kind wasn't
    # decided by how long each was held.
    assert combo.moves[0].duration_sec >= 1.0


def test_combo_touch_critique_mentions_touch_depth():
    records = _combo_records([(TOUCH_FRONT_LEVER_LANDMARKS, False), (FRONT_LEVER_LANDMARKS, False)])

    outcome = analyze_movement(records, movement_type_hint="combo")

    assert outcome is not None
    _, combo = outcome
    assert "Touch depth" in combo.moves[0].critique


def test_combo_each_move_has_a_one_sentence_critique():
    records = _combo_records([(TUCK_LANDMARKS, False), (FRONT_LEVER_LANDMARKS, False)])

    outcome = analyze_movement(records, movement_type_hint="combo")

    assert outcome is not None
    _, combo = outcome
    for move in combo.moves:
        assert move.critique
        assert isinstance(move.critique, str)


def test_combo_summary_lists_moves_in_order():
    records = _combo_records([(TUCK_LANDMARKS, False), (FRONT_LEVER_LANDMARKS, False)])

    outcome = analyze_movement(records, movement_type_hint="combo")

    assert outcome is not None
    _, combo = outcome
    assert "2-move combo" in combo.summary
    assert combo.summary.index("tuck") < combo.summary.index("full")


def test_no_movement_type_hint_never_infers_combo():
    # combo is only ever entered explicitly - auto-detection stays
    # static-then-dynamic, per analyze_movement's docstring.
    records = _combo_records([(TUCK_LANDMARKS, False), (FRONT_LEVER_LANDMARKS, False)])

    outcome = analyze_movement(records)

    assert outcome is not None
    kind, _ = outcome
    assert kind != "combo"


def test_combo_progression_hint_applies_to_every_move():
    # A real combo clip's every move shared the same progression
    # (straddle planche push-up into straddle planche press), but leg
    # tracking was too noisy for geometry to catch it correctly on either
    # move. The hint should apply uniformly across all detected moves.
    records = _combo_records([(TUCK_LANDMARKS, False), (FRONT_LEVER_LANDMARKS, False)])

    outcome = analyze_movement(records, movement_type_hint="combo", progression_hint="straddle")

    assert outcome is not None
    _, combo = outcome
    assert len(combo.moves) == 2
    assert combo.moves[0].progression == "straddle"
    assert combo.moves[1].progression == "straddle"
