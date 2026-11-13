"""Exact rational data model for the 30-note music box arranger."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path
import sys
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from inspect_notation_midi import (  # noqa: E402
    ALLOWED_MIDI,
    PITCH_NAMES,
    pitch_name,
    serialize_rationals,
)


class UnsupportedScoreError(ValueError):
    """Raised when an input or target score contains unsupported or ambiguous structures."""

    def __init__(self, message: str, issues: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.issues: list[dict[str, Any]] = issues or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.message,
            "issues": serialize_rationals(self.issues),
        }


def parse_fraction(value: F | int | str) -> F:
    if isinstance(value, F):
        return value
    return F(value)


@dataclass(frozen=True)
class SourceRef:
    """Exact provenance link back to an original source score notehead."""

    part: str
    measure: str
    measure_index: int
    note_index: int
    onset_q: F | None = None
    duration_q: F | None = None
    voice: str = "1"
    staff: str = "1"
    midi: int | None = None
    pitch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "measure": self.measure,
            "measure_index": self.measure_index,
            "note_index": self.note_index,
            "onset_q": str(self.onset_q) if self.onset_q is not None else None,
            "duration_q": str(self.duration_q) if self.duration_q is not None else None,
            "voice": self.voice,
            "staff": self.staff,
            "midi": self.midi,
            "pitch": self.pitch,
        }


@dataclass
class NoteEvent:
    """Independent sounding strike event using exact rational quarter-note timing."""

    id: str
    midi: int
    onset_q: F
    duration_q: F
    role: str
    source_refs: list[SourceRef] = field(default_factory=list)
    part: str = "P1"
    voice: str = "1"
    staff: str = "1"
    original_midi: int | None = None
    octave_adjusted: bool = False
    adjustment_reason: str | None = None
    duration_clamped_from_q: F | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "midi": self.midi,
            "pitch": pitch_name(self.midi),
            "onset_q": str(self.onset_q),
            "duration_q": str(self.duration_q),
            "role": self.role,
            "part": self.part,
            "voice": self.voice,
            "staff": self.staff,
            "original_midi": self.original_midi,
            "original_pitch": pitch_name(self.original_midi) if self.original_midi is not None else None,
            "octave_adjusted": self.octave_adjusted,
            "adjustment_reason": self.adjustment_reason,
            "duration_clamped_from_q": str(self.duration_clamped_from_q) if self.duration_clamped_from_q is not None else None,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
        }


@dataclass(frozen=True)
class MeasureSpan:
    """Linear timeline span for a single measure."""

    measure_index: int
    measure: str
    start_q: F
    end_q: F
    length_q: F
    nominal_q: F
    beats: int = 4
    beat_type: int = 4
    implicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure_index": self.measure_index,
            "measure": self.measure,
            "start_q": str(self.start_q),
            "end_q": str(self.end_q),
            "length_q": str(self.length_q),
            "nominal_q": str(self.nominal_q),
            "time_signature": f"{self.beats}/{self.beat_type}",
            "beats": self.beats,
            "beat_type": self.beat_type,
            "implicit": self.implicit,
        }


@dataclass(frozen=True)
class TempoEvent:
    """Explicit quarter-note tempo marking on the exact rational timeline."""

    onset_q: F
    quarter_bpm: F
    source: str = "score"

    def to_dict(self) -> dict[str, Any]:
        return {
            "onset_q": str(self.onset_q),
            "quarter_bpm": str(self.quarter_bpm),
            "source": self.source,
        }


@dataclass
class ScoreIR:
    """Intermediate representation shared across import, arrangement, export, and validation."""

    parts: list[str]
    part_names: dict[str, str]
    events: list[NoteEvent]
    tempos: list[TempoEvent]
    measures: list[MeasureSpan]
    total_q: F
    source_path: str | None = None
    source_sha256: str | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)
    unsupported_issues: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "parts": list(self.parts),
            "part_names": dict(self.part_names),
            "total_q": str(self.total_q),
            "measure_count": len(self.measures),
            "event_count": len(self.events),
            "measures": [m.to_dict() for m in self.measures],
            "tempos": [t.to_dict() for t in self.tempos],
            "events": [e.to_dict() for e in self.events],
            "warnings": serialize_rationals(self.warnings),
            "unsupported_issues": serialize_rationals(self.unsupported_issues),
            "metadata": serialize_rationals(self.metadata),
        }
