"""Plain-English explanation of a recommendation.

Pure text generation — this module must never import any rendering code, and
the rendered explanation itself only feeds the final Recommendation, never
the engine's math.
"""

from typing import List, Tuple

from .models import Recommendation
from .simulation import CandidateResult


def explain(recommendation: Recommendation, shot, best: CandidateResult) -> str:
    """Build a short human-readable paragraph for a recommendation.

    Rounding is deliberate: prose rounds expected strokes to 1 decimal and
    hazard probabilities to whole percent (they're already rounded to 0.05
    upstream). JSON keeps full precision.
    """
    top: List[Tuple[str, float]] = sorted(
        recommendation.hazard_probabilities.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_text = ", ".join(f"{region}: {pct:.0%}" for region, pct in top[:3])

    text = (
        f"Recommended: {recommendation.recommended_club} aimed at "
        f"({recommendation.recommended_target.x:.0f}, {recommendation.recommended_target.y:.0f}) — "
        f"expected {recommendation.expected_strokes:.1f} strokes. "
        f"Actual decision ({shot.club} at target) expected "
        f"{recommendation.actual_expected_strokes:.1f} strokes, a difference of "
        f"{recommendation.decision_cost:+.1f} strokes. "
        f"Top landing outcomes: {top_text}."
    )
    if recommendation.confidence == "low":
        text += (
            " Note: this recommendation is based on a limited sample of prior shots "
            "for one or both clubs compared; treat it as indicative, not precise."
        )
    return text
