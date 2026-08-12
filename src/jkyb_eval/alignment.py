"""Target extraction using bidirectional reference-context alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class TaggedParts:
    prefix: str
    target: str
    suffix: str


@dataclass(frozen=True)
class TargetAlignment:
    value: str
    start: int
    end: int
    status: str
    context_edit_distance: int
    context_error_rate: float
    ambiguous: bool
    candidate_count: int


def split_tagged(value: str) -> TaggedParts:
    if value.count("<") != 1 or value.count(">") != 1:
        raise ValueError("Tagged yomi must contain exactly one <...> target")
    opening = value.index("<")
    closing = value.index(">")
    if opening >= closing:
        raise ValueError("Tagged yomi has malformed target tags")
    return TaggedParts(
        prefix=value[:opening],
        target=value[opening + 1 : closing],
        suffix=value[closing + 1 :],
    )


def _prefix_edit_costs(pattern: str, value: str) -> list[int]:
    """Return ED(pattern, value[:i]) for every boundary i."""
    previous = list(range(len(value) + 1))
    for pattern_index, pattern_character in enumerate(pattern, start=1):
        current = [pattern_index]
        for value_index, value_character in enumerate(value, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[value_index] + 1,
                    previous[value_index - 1] + (pattern_character != value_character),
                )
            )
        previous = current
    return previous


def _search_space(
    parts: TaggedParts,
    prediction: str,
) -> tuple[list[int], list[int], list[int], list[int], int, int]:
    left_costs = _prefix_edit_costs(parts.prefix, prediction)
    reversed_right_costs = _prefix_edit_costs(parts.suffix[::-1], prediction[::-1])
    right_costs = [
        reversed_right_costs[len(prediction) - boundary]
        for boundary in range(len(prediction) + 1)
    ]

    # The split at the reference boundary is always available, so its context
    # cost is an upper bound on the best split. A boundary whose individual
    # context cost exceeds that bound cannot be part of a winning pair. This
    # keeps runaway predictions proportional to the reference context instead
    # of considering every quadratic pair of prediction boundaries.
    natural_start = min(len(parts.prefix), len(prediction))
    natural_end = max(natural_start, len(prediction) - len(parts.suffix))
    bound = left_costs[natural_start] + right_costs[natural_end]
    starts = [index for index, cost in enumerate(left_costs) if cost <= bound]
    ends = [index for index, cost in enumerate(right_costs) if cost <= bound]
    return left_costs, right_costs, starts, ends, natural_start, natural_end


def _result(
    *,
    parts: TaggedParts,
    prediction: str,
    pair: tuple[int, int],
    context_distance: int,
    candidate_count: int,
) -> TargetAlignment:
    start, end = pair
    target = prediction[start:end]
    if not target:
        status = "empty_mapped_span"
    elif candidate_count > 1:
        status = "ambiguous"
    else:
        status = "mapped"

    context_length = len(parts.prefix) + len(parts.suffix)
    return TargetAlignment(
        value=target,
        start=start,
        end=end,
        status=status,
        context_edit_distance=context_distance,
        context_error_rate=context_distance / max(1, context_length),
        ambiguous=candidate_count > 1,
        candidate_count=candidate_count,
    )


def align_target(
    tagged_reference: str,
    prediction: str,
    *,
    accepted_lengths: Iterable[int],
) -> TargetAlignment:
    """Extract the prediction span bracketed by the best prefix/suffix alignment."""
    parts = split_tagged(tagged_reference)
    expected_lengths = sorted(set(accepted_lengths))
    if not expected_lengths or any(length <= 0 for length in expected_lengths):
        raise ValueError("accepted_lengths must contain positive lengths")

    left_costs, right_costs, starts, ends, natural_start, natural_end = _search_space(
        parts, prediction
    )

    # Ascending order means the first pair to reach the best score is already
    # the smallest one, so the winner is tracked instead of collected.
    best_semantic_score: tuple[int, int, int] | None = None
    best_pair = (natural_start, natural_end)
    candidate_count = 0
    for start in starts:
        for end in ends:
            if end < start:
                continue
            target_length = end - start
            semantic_score = (
                left_costs[start] + right_costs[end],
                min(abs(target_length - length) for length in expected_lengths),
                abs(start - len(parts.prefix))
                + abs((len(prediction) - end) - len(parts.suffix)),
            )
            if best_semantic_score is None or semantic_score < best_semantic_score:
                best_semantic_score = semantic_score
                best_pair = (start, end)
                candidate_count = 1
            elif semantic_score == best_semantic_score:
                candidate_count += 1

    assert best_semantic_score is not None  # the natural pair always qualifies
    return _result(
        parts=parts,
        prediction=prediction,
        pair=best_pair,
        context_distance=best_semantic_score[0],
        candidate_count=candidate_count,
    )


def align_candidate(
    tagged_reference: str,
    prediction: str,
    *,
    candidate: str,
) -> TargetAlignment:
    """Align one accepted target reading against its surrounding context.

    Each natural or marginal reading gets its own alignment. This matters when
    alternatives divide a word differently and therefore imply different
    target lengths. Context remains the primary boundary signal; candidate
    similarity only resolves boundaries with the same context cost.
    """
    if not candidate:
        raise ValueError("candidate must be non-empty")

    parts = split_tagged(tagged_reference)
    left_costs, right_costs, starts, ends, _, _ = _search_space(parts, prediction)

    # With no context on one side, that sentence edge is the only defensible
    # boundary. This prevents an accepted reading elsewhere in the prediction
    # from rescuing a target that begins or ends the sentence.
    if not parts.prefix:
        starts = [0]
    if not parts.suffix:
        ends = [len(prediction)]

    best_semantic_score: tuple[int, int, int] | None = None
    best_pair = (starts[0], ends[-1])
    candidate_count = 0
    for start in starts:
        for end in ends:
            if end < start:
                continue
            semantic_score = (
                left_costs[start] + right_costs[end],
                Levenshtein.distance(candidate, prediction[start:end]),
                abs(start - len(parts.prefix))
                + abs((len(prediction) - end) - len(parts.suffix)),
            )
            if best_semantic_score is None or semantic_score < best_semantic_score:
                best_semantic_score = semantic_score
                best_pair = (start, end)
                candidate_count = 1
            elif semantic_score == best_semantic_score:
                candidate_count += 1

    assert best_semantic_score is not None  # at least one ordered pair exists
    return _result(
        parts=parts,
        prediction=prediction,
        pair=best_pair,
        context_distance=best_semantic_score[0],
        candidate_count=candidate_count,
    )
