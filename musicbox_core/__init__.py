"""30-note music box MVP-1 arrangement core package."""
from __future__ import annotations

from musicbox_core.model import (
    ALLOWED_MIDI,
    MeasureSpan,
    NoteEvent,
    ScoreIR,
    SourceRef,
    TempoEvent,
    UnsupportedScoreError,
    pitch_name,
    serialize_rationals,
)
from musicbox_core.importer import load_score
from musicbox_core.arrange import analyze_score, arrange_score
from musicbox_core.exporter import export_variant
from musicbox_core.validation import validate_variant

__all__ = [
    "ALLOWED_MIDI",
    "MeasureSpan",
    "NoteEvent",
    "ScoreIR",
    "SourceRef",
    "TempoEvent",
    "UnsupportedScoreError",
    "analyze_score",
    "arrange_score",
    "export_variant",
    "load_score",
    "pitch_name",
    "serialize_rationals",
    "validate_variant",
]
