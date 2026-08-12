"""Validated in-memory representations of benchmark inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import READING_CATEGORIES


def _strip_single_tag(value: str, *, field: str) -> str:
    if value.count("<") != 1 or value.count(">") != 1:
        raise ValueError(f"{field} must contain exactly one <...> tag")
    if value.index("<") >= value.index(">"):
        raise ValueError(f"{field} has malformed target tags")
    return value.replace("<", "").replace(">", "")


@dataclass(frozen=True)
class BenchmarkRow:
    key: str
    text: str
    tagged_text: str
    yomi: str
    tagged_yomi: str
    reading_category: str
    natural: tuple[str, ...]
    marginal: tuple[str, ...]
    source: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, location: str) -> BenchmarkRow:
        required = ("key", "text", "tagged_text", "yomi", "tagged_yomi")
        values: dict[str, str] = {}
        for field in required:
            value = raw.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{location}: {field} must be a non-empty string")
            values[field] = value

        readings = raw.get("readings")
        if not isinstance(readings, dict):
            raise ValueError(f"{location}: readings must be an object")

        def reading_list(name: str, *, required_nonempty: bool) -> tuple[str, ...]:
            items = readings.get(name)
            if not isinstance(items, list) or not all(
                isinstance(item, str) and item for item in items
            ):
                raise ValueError(f"{location}: readings.{name} must be a string list")
            if required_nonempty and not items:
                raise ValueError(f"{location}: readings.{name} must not be empty")
            if len(items) != len(set(items)):
                raise ValueError(f"{location}: readings.{name} contains duplicates")
            return tuple(items)

        natural = reading_list("natural", required_nonempty=True)
        marginal = reading_list("marginal", required_nonempty=False)
        overlap = set(natural) & set(marginal)
        if overlap:
            raise ValueError(
                f"{location}: readings occur in both natural and marginal: "
                f"{sorted(overlap)!r}"
            )

        if (
            _strip_single_tag(values["tagged_text"], field="tagged_text")
            != values["text"]
        ):
            raise ValueError(f"{location}: tagged_text does not reduce to text")
        if (
            _strip_single_tag(values["tagged_yomi"], field="tagged_yomi")
            != values["yomi"]
        ):
            raise ValueError(f"{location}: tagged_yomi does not reduce to yomi")

        source = raw.get("source", "")
        if not isinstance(source, str):
            raise ValueError(f"{location}: source must be a string")

        reading_category = raw.get("reading_category")
        if reading_category not in READING_CATEGORIES:
            raise ValueError(
                f"{location}: reading_category must be one of {READING_CATEGORIES!r}"
            )

        return cls(
            key=values["key"],
            text=values["text"],
            tagged_text=values["tagged_text"],
            yomi=values["yomi"],
            tagged_yomi=values["tagged_yomi"],
            reading_category=reading_category,
            natural=natural,
            marginal=marginal,
            source=source,
        )


@dataclass(frozen=True)
class Prediction:
    key: str
    value: str
