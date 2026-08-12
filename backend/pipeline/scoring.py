"""Form-quality scoring for a detected hold window.

Three criteria are scored 0-100, thresholds grounded in the two real sample
clips (see conversation/plan history for the underlying numbers):
  - arm_lockout: elbow straightness (shoulder-elbow-wrist angle)
  - hip_shoulder_alignment: body-line straightness (hip deviation from the
    shoulder-ankle line), with a sag/pike direction label
  - hold_stability: how still the body stayed during the window (median
    frame-to-frame joint displacement, reusing hold_detection's signal)

Scapular position (protraction vs retraction) was in the original scope but
is deliberately NOT scored here - see SCAPULAR_POSITION_NOTE. Every proxy
computed from the available 2D side-view landmarks either came out
numerically degenerate (shoulder-width, used as a normalization reference,
collapses to near-zero from a side view since left/right shoulders nearly
overlap in the image) or measured something else entirely (arm-to-torso
angle reflects the move's geometry, not scapular engagement, and was stable
but not meaningful as a quality signal). Rather than report a fabricated
score, this is surfaced explicitly as unmeasurable. "hold_stability" - a
real, data-grounded signal not in the original three-item list - fills the
third scoring slot instead, under the vision doc's "any other form
breakdowns worth detecting" allowance.
"""
import statistics
from dataclasses import dataclass
from typing import Optional

from pipeline.feedback import build_feedback
from pipeline.geometry import distance, joint_angle
from pipeline.hold_detection import DEFAULT_STABILITY_THRESHOLD, _frame_displacement

SCAPULAR_POSITION_NOTE = (
    "Scapular position (protraction vs retraction) is not scored. It can't be "
    "reliably measured from a single side-view 2D camera with the current "
    "landmark set - true shoulder-blade position is primarily visible from a "
    "front or 3/4 camera angle. Reporting this as unmeasurable rather than a "
    "fabricated score."
)

# Tunable constants, grounded against the two real sample clips.
#
# arm_lockout and hip_shoulder_alignment were originally scored against a
# literal geometric ideal (180 degrees dead-straight elbow, 0 deviation from
# a perfectly straight hip line). Checked against every real hold clip on
# hand (9 clips, elbow angles 150.7-170.1 degrees, hip deviations 0.029-1.17):
# no real hold - including one independently judged "almost perfect, full
# lockout" by eye, whose *reliably-tracked* near-side elbow measured 160.1
# degrees and hip deviation measured 0.036 - ever approached those literal
# ideals. A single 2D side-view camera's practical measurement ceiling (skin/
# muscle occlusion of the true joint center, camera angle, capture
# resolution) means real excellent form reads as "165ish degrees" and "~0.03
# deviation", not literally 180/0. Penalizing relative to the literal ideal
# made a genuinely excellent hold score in the high-70s/low-80s and a small,
# likely-within-measurement-noise deviation read as "sagging" when the
# athlete's hips were visibly dead straight. Both criteria now score against
# a *practical* reference point (what real excellent form actually measures
# at) rather than the unreachable geometric ideal, while keeping real bad
# form (a 30+ degree bend, genuine multi-tenths hip sag) clearly penalized -
# see PRACTICAL_LOCKOUT_REFERENCE_DEG / PRACTICAL_HIP_ALIGNMENT_REFERENCE and
# the reduced per-unit penalties below.
# PRACTICAL_LOCKOUT_REFERENCE_DEG was originally 165.0, chosen as "what real
# excellent form measures at". That reference doubled as a hard ceiling
# (anything >= it scored a flat 100), which meant every clip in the
# 165-170 degree range - a real, meaningfully-sized band, not measurement
# noise - scored identically to a hypothetical perfect 180 degree lockout.
# A real full front lever clip (166.0 degrees, median-tracked) came back
# 96.6/100 overall as a result, which read as too generous once actually
# compared side by side against the source footage (independently judged
# closer to 92) - the ceiling was doing the over-crediting, not a bug in
# any single criterion. Raised to 175.0 so the 165-170 band (and the few
# degrees past it) still separates on a genuine lockout difference instead
# of flatlining; still well short of the literal 180 ideal this system
# deliberately moved away from, and still gives a real ~170 degree lockout
# (the best angle across every real sample clip on hand) a high-90s score,
# not a punishing one. HOLD_STABILITY_SCALE bumped proportionally (+15%)
# for the same reason - both were re-checked across every real sample clip
# on hand (see the sweep in git history / conversation) to confirm this
# shifts scores down a modest, consistent amount without collapsing any
# clip's relative ranking or its qualitative label.
PRACTICAL_LOCKOUT_REFERENCE_DEG = 175.0  # elbow angle a genuinely excellent lockout measures at
ARM_LOCKOUT_DEG_PENALTY = 1.0  # points lost per degree of elbow bend below the practical reference
STRAIGHT_BODY_LINE_DEVIATION_THRESHOLD = 0.04  # was 0.03 - see recalibration note above
HIP_ALIGNMENT_DEVIATION_PENALTY = 175.0  # points lost per unit of normalized hip deviation
HOLD_STABILITY_SCALE = 46.0  # points lost per multiple of DEFAULT_STABILITY_THRESHOLD
LOW_CONFIDENCE_VISIBILITY_FLOOR = 0.6


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _midpoint(a: dict, b: dict) -> dict:
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2}


def _min_visibility(records: list[dict], joints: tuple) -> float:
    """Average visibility across the given joints.

    Originally took the strict minimum across every joint including both
    left and right sides, on the theory that a single occluded joint should
    flag low confidence. Real footage showed this always fires: every
    side-view hold has one arm/leg partially behind the body, so its far-side
    elbow/wrist/ankle visibility sits at 0.05-0.4 in every real clip checked
    (IMG_8553, IMG_1231, IMG_1233, IMG_1234) regardless of how well the hold
    is actually tracked. That made arm_lockout/hip_shoulder_alignment come
    back "low confidence" on essentially every real video, which then halved
    their weight in the overall score and compressed the headline number
    toward hold_stability alone - the reported "scores don't vary enough"
    symptom. Averaging (which comfortably clears the 0.6 floor at 0.7-0.96 on
    the same real clips) reflects "is there usable signal here" without being
    vetoed by a limb that's expected to be occluded by camera geometry.
    """
    values = [r["landmarks"][j]["visibility"] for r in records for j in joints if j in r["landmarks"]]
    return statistics.mean(values) if values else 0.0


def _confidence(min_visibility: float) -> str:
    return "high" if min_visibility >= LOW_CONFIDENCE_VISIBILITY_FLOOR else "low"


def _body_line_deviation(shoulder: dict, hip: dict, ankle: dict) -> Optional[float]:
    """Signed, normalized deviation of the hip from the shoulder-ankle line.

    Positive = hip below the line (sag), negative = above (pike). Computed
    via linear interpolation of the expected hip height at the hip's actual
    x-position, rather than a raw cross-product sign, so the result doesn't
    flip depending on which horizontal direction the athlete faces in shot.
    """
    dx = ankle["x"] - shoulder["x"]
    if dx == 0:
        return None
    t = (hip["x"] - shoulder["x"]) / dx
    expected_hip_y = shoulder["y"] + t * (ankle["y"] - shoulder["y"])
    line_len = distance(shoulder, ankle)
    if line_len == 0:
        return None
    return (hip["y"] - expected_hip_y) / line_len


@dataclass
class CriterionScore:
    score: float
    label: str
    confidence: str
    detail: dict


def score_arm_lockout(records: list[dict]) -> CriterionScore:
    """Scores elbow straightness against the practical reference a real
    excellent hold measures at (see PRACTICAL_LOCKOUT_REFERENCE_DEG note),
    not the unreachable literal 180-degree ideal.
    """
    angles = []
    for r in records:
        lm = r["landmarks"]
        left = joint_angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
        right = joint_angle(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])
        angles.append((left + right) / 2)

    avg_angle = statistics.median(angles)
    score = _clamp(100 - max(0.0, PRACTICAL_LOCKOUT_REFERENCE_DEG - avg_angle) * ARM_LOCKOUT_DEG_PENALTY)

    if avg_angle >= 170:
        label = "excellent lockout"
    elif avg_angle >= 155:
        label = "good lockout, slight bend"
    elif avg_angle >= 130:
        label = "noticeable elbow bend"
    else:
        label = "significant elbow bend"

    joints = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
    confidence = _confidence(_min_visibility(records, joints))

    return CriterionScore(
        score=round(score, 1),
        label=label,
        confidence=confidence,
        detail={"avg_elbow_angle_deg": round(avg_angle, 1)},
    )


def score_hip_shoulder_alignment(records: list[dict]) -> CriterionScore:
    """Scores body-line straightness and reports sag vs pike direction."""
    deviations = []
    for r in records:
        lm = r["landmarks"]
        shoulder = _midpoint(lm["left_shoulder"], lm["right_shoulder"])
        hip = _midpoint(lm["left_hip"], lm["right_hip"])
        ankle = _midpoint(lm["left_ankle"], lm["right_ankle"])
        dev = _body_line_deviation(shoulder, hip, ankle)
        if dev is not None:
            deviations.append(dev)

    joints = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle")
    confidence = _confidence(_min_visibility(records, joints))

    if not deviations:
        return CriterionScore(score=0.0, label="could not measure body line", confidence="low", detail={})

    avg_dev = statistics.median(deviations)
    score = _clamp(100 - abs(avg_dev) * HIP_ALIGNMENT_DEVIATION_PENALTY)

    if abs(avg_dev) < STRAIGHT_BODY_LINE_DEVIATION_THRESHOLD:
        direction = "straight"
        label = "straight body line"
    elif avg_dev > 0:
        direction = "sagging"
        label = "hips sagging"
    else:
        direction = "piking"
        label = "hips piking"

    return CriterionScore(
        score=round(score, 1),
        label=label,
        confidence=confidence,
        detail={"body_line_deviation": round(avg_dev, 3), "direction": direction},
    )


def score_hold_stability(records: list[dict]) -> CriterionScore:
    """Scores how still the body stayed during the window."""
    disps = []
    for prev, curr in zip(records, records[1:]):
        d = _frame_displacement(prev["landmarks"], curr["landmarks"])
        if d is not None:
            disps.append(d)

    if not disps:
        return CriterionScore(score=0.0, label="could not measure stability", confidence="low", detail={})

    median_disp = statistics.median(disps)
    score = _clamp(100 - (median_disp / DEFAULT_STABILITY_THRESHOLD) * HOLD_STABILITY_SCALE)

    if score >= 85:
        label = "very stable hold"
    elif score >= 65:
        label = "reasonably stable, some movement"
    else:
        label = "visibly unstable/shaking"

    confidence = "high" if len(disps) >= 3 else "low"

    return CriterionScore(
        score=round(score, 1),
        label=label,
        confidence=confidence,
        detail={"median_displacement": round(median_disp, 4)},
    )


LOW_CONFIDENCE_WEIGHT = 0.5  # a low-confidence criterion still shows in the
# breakdown, but shouldn't fully sway the headline number - a single badly
# occluded joint (e.g. IMG_0164's known ankle-tracking issue) can otherwise
# drag "overall form score" down in a way that reflects tracking noise, not
# actual form.


@dataclass
class FormReport:
    overall_score: float
    overall_confidence: str  # "high" | "mixed" | "low"
    criteria: dict  # name -> CriterionScore
    strengths: list  # detailed, specific - see pipeline/feedback.py
    refine: list  # solid but improvable
    weaknesses: list  # concrete, actionable
    summary: str
    scapular_position_note: str


def compute_form_report(records: list[dict], subject_label: str = "hold") -> Optional[FormReport]:
    """records: pose records restricted to the detected hold window.

    subject_label describes what's being scored (e.g. "full front lever",
    "planche push-up") - used to phrase the summary naturally. Callers that
    don't know the specific move/progression yet can leave the generic
    default.
    """
    if not records:
        return None

    criteria = {
        "arm_lockout": score_arm_lockout(records),
        "hip_shoulder_alignment": score_hip_shoulder_alignment(records),
        "hold_stability": score_hold_stability(records),
    }

    weights = [1.0 if c.confidence == "high" else LOW_CONFIDENCE_WEIGHT for c in criteria.values()]
    weighted_scores = [c.score * w for c, w in zip(criteria.values(), weights)]
    overall = round(sum(weighted_scores) / sum(weights), 1)

    confidences = {c.confidence for c in criteria.values()}
    if confidences == {"high"}:
        overall_confidence = "high"
    elif confidences == {"low"}:
        overall_confidence = "low"
    else:
        overall_confidence = "mixed"

    fb = build_feedback(criteria, overall, subject_label)

    return FormReport(
        overall_score=overall,
        overall_confidence=overall_confidence,
        criteria=criteria,
        strengths=fb.strengths,
        refine=fb.refine,
        weaknesses=fb.weaknesses,
        summary=fb.summary,
        scapular_position_note=SCAPULAR_POSITION_NOTE,
    )
