"""Independent MusicXML duration/notation and MIDI onset diagnostics for M0.

All time calculations use exact rational numbers (fractions.Fraction).
Unsupported constructs (missing/unknown note types, grace notes, non-PPQ SMPTE
MIDI timebases, unexpanded navigation, etc.) are reported explicitly instead of
guessing defaults.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
ALLOWED_MIDI = frozenset([53, 55, 60, 62, 64, 65, 67, 69, 70, 71, *range(72, 90), 91, 93])
NOTE_TYPES: dict[str, F] = {
    "maxima": F(32),
    "long": F(16),
    "breve": F(8),
    "whole": F(4),
    "half": F(2),
    "quarter": F(1),
    "eighth": F(1, 2),
    "16th": F(1, 4),
    "32nd": F(1, 8),
    "64th": F(1, 16),
    "128th": F(1, 32),
    "256th": F(1, 64),
    "512th": F(1, 128),
    "1024th": F(1, 256),
}


def pitch_name(midi: int | F) -> str:
    frac = F(midi)
    if frac.denominator != 1:
        return f"MIDI:{frac}"
    m = int(frac)
    return f"{PITCH_NAMES[m % 12]}{m // 12 - 1}"


def serialize_rationals(value: Any) -> Any:
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize_rationals(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_rationals(v) for v in value]
    if isinstance(value, set):
        return [serialize_rationals(v) for v in sorted(value, key=str)]
    return value


def _read_xml_bytes(source: str | bytes | Path) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, Path):
        if source.suffix.lower() == ".mxl":
            with zipfile.ZipFile(source) as zf:
                container = ET.fromstring(zf.read("META-INF/container.xml"))
                roots = [e for e in container.iter() if e.tag.split("}")[-1] == "rootfile"]
                preferred = [
                    e for e in roots if e.get("media-type") == "application/vnd.recordare.musicxml+xml"
                ]
                if not roots:
                    raise ValueError("MXL has no rootfile")
                target = (preferred or roots)[0].get("full-path")
                if zf.getinfo(target).file_size > 50_000_000:
                    raise ValueError("MXL rootfile exceeds 50 MB")
                return zf.read(target)
        return source.read_bytes()
    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("<"):
            return source.encode("utf-8")
        return _read_xml_bytes(Path(source))
    raise TypeError(f"Unsupported xml_source type: {type(source).__name__}")


def _read_midi_bytes(source: bytes | Path | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (Path, str)):
        return Path(source).read_bytes()
    raise TypeError(f"Unsupported midi_source type: {type(source).__name__}")


def _parse_musicxml(xml_source: str | bytes | Path, input_id: str) -> dict[str, Any]:
    try:
        raw_bytes = _read_xml_bytes(xml_source)
        root = ET.fromstring(raw_bytes)
    except (ET.ParseError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return {"status": "unsupported", "reason": f"xml_parse_error: {exc}", "input_id": input_id}

    for elem in root.iter():
        elem.tag = elem.tag.split("}")[-1]

    if root.tag != "score-partwise":
        return {"status": "unsupported", "reason": f"unsupported_score_type: {root.tag}", "input_id": input_id}

    notes: list[dict[str, Any]] = []
    pitched_events: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    transpose_events: list[dict[str, Any]] = []
    part_measures: list[list[dict[str, Any]]] = []
    parts: list[str] = []

    for pi, part in enumerate(root.findall("part")):
        part_id = part.get("id", str(pi + 1))
        parts.append(part_id)
        div = F(1)
        has_divisions = False
        nominal = F(4)
        transpositions: dict[str, F] = {"all": F(0)}
        pmeasures: list[dict[str, Any]] = []

        for mi, measure in enumerate(part.findall("measure")):
            measure_num = measure.get("number", str(mi + 1))
            loc = {"part": part_id, "part_index": pi, "measure": measure_num, "measure_index": mi}
            cursor = F(0)
            extent = F(0)
            last_on: F | None = None
            note_idx_in_measure = 0

            for item in measure:
                if item.tag == "attributes":
                    div_elem = item.find("divisions")
                    if div_elem is not None and div_elem.text is not None:
                        try:
                            div = F(div_elem.text.strip())
                        except ValueError:
                            blockers.append({"code": "invalid_divisions", **loc})
                            div = F(0)
                        if div <= 0:
                            blockers.append({"code": "invalid_divisions", **loc})
                        else:
                            has_divisions = True

                    time_elem = item.find("time")
                    if time_elem is not None:
                        beats = time_elem.findall("beats")
                        beat_types = time_elem.findall("beat-type")
                        if len(beats) != len(beat_types) or not beats:
                            blockers.append({"code": "unsupported_meter", **loc})
                        else:
                            try:
                                nominal = sum(
                                    (
                                        sum(F(n.strip()) for n in (b.text or "").split("+")) * 4 / F((t.text or "").strip())
                                        for b, t in zip(beats, beat_types)
                                    ),
                                    F(0),
                                )
                            except (ValueError, ZeroDivisionError):
                                blockers.append({"code": "unsupported_meter", **loc})

                    for tr in item.findall("transpose"):
                        if cursor != 0:
                            blockers.append({"code": "unresolved_midmeasure_transposition", **loc, "measure_offset_q": cursor})
                        if tr.find("double") is not None:
                            blockers.append({"code": "unresolved_transposition_doubling", **loc})
                        chromatic = F(tr.findtext("chromatic", "0").strip())
                        oct_change = F(tr.findtext("octave-change", "0").strip())
                        amount = chromatic + 12 * oct_change
                        if tr.get("number") is None:
                            transpositions = {"all": amount}
                        else:
                            transpositions[tr.get("number")] = amount
                        transpose_events.append(
                            {
                                "part": part_id,
                                "measure": measure_num,
                                "staff": tr.get("number", "all"),
                                "semitones": amount,
                            }
                        )

                elif item.tag in ("backup", "forward"):
                    if not has_divisions or div <= 0:
                        blockers.append({"code": "missing_or_invalid_divisions_for_cursor", **loc})
                        continue
                    amount = F((item.findtext("duration") or "0").strip()) / div
                    cursor += amount if item.tag == "forward" else -amount
                    if cursor < 0:
                        blockers.append({"code": "negative_cursor", **loc})
                    extent = max(extent, cursor)
                    last_on = None

                elif item.tag == "note":
                    note_idx_in_measure += 1
                    if any(F(item.get(k, "0")) != 0 for k in ("attack", "release")):
                        blockers.append({"code": "unresolved_performance_timing", **loc, "measure_offset_q": cursor})
                    for orn_tag in ("tremolo", "trill-mark", "mordent", "inverted-mordent", "turn", "inverted-turn"):
                        if item.find(f".//{orn_tag}") is not None:
                            blockers.append({"code": "unresolved_ornament", "ornament": orn_tag, **loc})

                    grace = item.find("grace") is not None
                    chord = item.find("chord") is not None
                    rest_elem = item.find("rest")
                    pitch_elem = item.find("pitch")
                    unpitched_elem = item.find("unpitched")

                    dur_text = item.findtext("duration")
                    actual_q: F | None = None
                    if not grace and has_divisions and div > 0 and dur_text is not None:
                        try:
                            actual_q = F(dur_text.strip()) / div
                        except ValueError:
                            actual_q = None

                    if grace:
                        blockers.append({"code": "unresolved_grace", **loc, "measure_offset_q": cursor})
                        onset_q = cursor
                    else:
                        if chord and last_on is None:
                            blockers.append({"code": "orphan_chord", **loc})
                        onset_q = last_on if (chord and last_on is not None) else cursor
                        if actual_q is not None:
                            if not chord:
                                last_on = cursor
                                cursor += actual_q
                            extent = max(extent, cursor, onset_q + actual_q)

                    kind = "rest" if rest_elem is not None else ("note" if pitch_elem is not None else ("unpitched" if unpitched_elem is not None else "unknown"))
                    pitch_str: str | None = None
                    written_midi_val: F | None = None
                    midi_val: F | None = None
                    if pitch_elem is not None:
                        step = (pitch_elem.findtext("step") or "").strip()
                        alter = F((pitch_elem.findtext("alter") or "0").strip())
                        octave = int((pitch_elem.findtext("octave") or "0").strip())
                        written_midi_val = F(12 * (octave + 1) + STEPS[step]) + alter
                        staff = (item.findtext("staff") or "1").strip()
                        transposition = transpositions.get(staff, transpositions["all"])
                        midi_val = written_midi_val + transposition
                        pitch_str = pitch_name(midi_val)
                    else:
                        staff = (item.findtext("staff") or "1").strip()

                    ties = {t.get("type") for t in item.findall("tie") if t.get("type") is not None}
                    tieds = {t.get("type") for t in item.findall("./notations/tied") if t.get("type") is not None}
                    if any(t.get("time-only") for t in item.findall("tie")):
                        blockers.append({"code": "conditional_tie", **loc})

                    tm_elem = item.find("time-modification")
                    tm_info: dict[str, Any] | None = None
                    if tm_elem is not None:
                        tm_info = {
                            "actual_notes": tm_elem.findtext("actual-notes"),
                            "normal_notes": tm_elem.findtext("normal-notes"),
                            "normal_type": tm_elem.findtext("normal-type"),
                        }

                    rec = {
                        "input_id": input_id,
                        **loc,
                        "measure_offset_q": onset_q,
                        "note_index_in_measure": note_idx_in_measure,
                        "kind": kind,
                        "grace": grace,
                        "chord": chord,
                        "rest_measure": rest_elem.get("measure") if rest_elem is not None else None,
                        "divisions": div if has_divisions else None,
                        "duration_raw": dur_text.strip() if dur_text is not None else None,
                        "actual_q": actual_q,
                        "nominal_measure_q": nominal,
                        "type": item.findtext("type").strip() if item.findtext("type") is not None else None,
                        "dots": len(item.findall("dot")),
                        "time_modification": tm_info,
                        "pitch": pitch_str,
                        "written_midi": written_midi_val,
                        "midi": midi_val,
                        "voice": (item.findtext("voice") or "1").strip(),
                        "staff": staff,
                        "ties": ties,
                        "tieds": tieds,
                        "order": len(notes),
                    }
                    notes.append(rec)
                    if pitch_elem is not None:
                        pitched_events.append(rec)

                elif item.tag in ("direction", "sound"):
                    if item.find("./direction-type/octave-shift") is not None:
                        blockers.append({"code": "unresolved_octave_shift", **loc})
                    sounds = [item] if item.tag == "sound" else item.findall("sound")
                    for sound in sounds:
                        if any(k in sound.attrib for k in ("dacapo", "dalsegno", "tocoda", "fine", "forward-repeat")):
                            blockers.append({"code": "unexpanded_navigation", **loc})
                        if sound.get("time-only"):
                            blockers.append({"code": "conditional_sound", **loc})
                elif item.tag == "barline":
                    if item.find("repeat") is not None or item.find("ending") is not None:
                        blockers.append({"code": "unexpanded_navigation", **loc})

            length_q = extent if extent > 0 else nominal
            pmeasures.append({**loc, "actual": extent, "length": length_q, "nominal": nominal})

        part_measures.append(pmeasures)

    count = max((len(pm) for pm in part_measures), default=0)
    if any(len(pm) != count for pm in part_measures):
        blockers.append({"code": "part_measure_count_mismatch"})

    starts: list[F] = []
    timeline: list[dict[str, Any]] = []
    cumulative = F(0)
    for mi in range(count):
        group = [pm[mi] for pm in part_measures if mi < len(pm)]
        if len({g["length"] for g in group}) > 1:
            blockers.append({"code": "part_measure_length_mismatch", "measure_index": mi})
        bar_len = max(g["length"] for g in group)
        starts.append(cumulative)
        timeline.append(
            {
                "measure": group[0]["measure"],
                "measure_index": mi,
                "start_q": cumulative,
                "end_q": cumulative + bar_len,
                "length_q": bar_len,
            }
        )
        cumulative += bar_len

    for rec in notes:
        mi = rec["measure_index"]
        rec["abs_q"] = starts[mi] + rec["measure_offset_q"] if mi < len(starts) else rec["measure_offset_q"]

    return {
        "status": "ok",
        "input_id": input_id,
        "parts": parts,
        "notes": notes,
        "pitched_events": pitched_events,
        "blockers": blockers,
        "transpose_events": transpose_events,
        "measure_timeline": timeline,
        "total_q": cumulative,
    }


def _evaluate_graphic_duration(rec: dict[str, Any]) -> tuple[str, F | None, str | None]:
    """Return (classification, notated_q, reason) for a parsed MusicXML <note> record."""
    if rec["grace"]:
        return ("unsupported", None, "grace_note_no_duration")
    if rec["divisions"] is None or rec["divisions"] <= 0:
        return ("unsupported", None, "missing_or_invalid_divisions")
    if rec["actual_q"] is None or rec["actual_q"] <= 0:
        return ("unsupported", None, "missing_or_invalid_duration")
    if rec["kind"] not in ("note", "rest"):
        return ("unsupported", None, f"unsupported_note_kind_{rec['kind']}")

    note_type = rec["type"]
    dots = rec["dots"]
    tm = rec["time_modification"]

    if rec["kind"] == "rest" and rec["rest_measure"] == "yes":
        if note_type is None or (note_type == "whole" and dots == 0 and tm is None):
            if rec["actual_q"] == rec["nominal_measure_q"]:
                return ("whole_measure_rest", rec["nominal_measure_q"], None)
            return ("mismatch", rec["nominal_measure_q"], "whole_measure_rest_duration_mismatch")

    if note_type is None:
        return ("unsupported", None, "missing_type")
    if note_type not in NOTE_TYPES:
        return ("unsupported", None, "unknown_type")

    base_q = NOTE_TYPES[note_type]
    dot_mult = sum((F(1, 2**i) for i in range(dots + 1)), F(0))
    tuplet_ratio = F(1)
    if tm is not None:
        try:
            act_n = int((tm["actual_notes"] or "").strip())
            norm_n = int((tm["normal_notes"] or "").strip())
        except ValueError:
            return ("unsupported", None, "invalid_time_modification")
        if act_n <= 0 or norm_n <= 0:
            return ("unsupported", None, "invalid_time_modification")
        tuplet_ratio = F(norm_n, act_n)

    notated_q = base_q * dot_mult * tuplet_ratio
    if rec["actual_q"] == notated_q:
        return ("consistent", notated_q, None)
    return ("mismatch", notated_q, "actual_duration_differs_from_graphic_notation")


def inspect_notation_durations(xml_source: str | bytes | Path, input_id: str = "inline") -> dict[str, Any]:
    """Compare MusicXML duration/divisions against graphic type, dots, and time-modification."""
    parsed = _parse_musicxml(xml_source, input_id)
    if parsed["status"] != "ok":
        return {
            "status": "unsupported",
            "input_id": input_id,
            "reason": parsed["reason"],
            "checked_items": 0,
            "consistent_count": 0,
            "whole_measure_rest_count": 0,
            "mismatch_count": 0,
            "unsupported_count": 1,
            "mismatches": [],
            "unsupported_items": [{"input_id": input_id, "reason": parsed["reason"]}],
        }

    consistent_count = 0
    whole_measure_rest_count = 0
    mismatches: list[dict[str, Any]] = []
    unsupported_items: list[dict[str, Any]] = []

    for rec in parsed["notes"]:
        classification, notated_q, reason = _evaluate_graphic_duration(rec)
        base_loc = {
            "input_id": input_id,
            "part": rec["part"],
            "part_index": rec["part_index"],
            "measure": rec["measure"],
            "measure_index": rec["measure_index"],
            "measure_offset_q": str(rec["measure_offset_q"]),
            "abs_q": str(rec["abs_q"]),
            "note_index_in_measure": rec["note_index_in_measure"],
            "kind": rec["kind"],
            "pitch": rec["pitch"],
        }
        if classification == "consistent":
            consistent_count += 1
        elif classification == "whole_measure_rest":
            whole_measure_rest_count += 1
            consistent_count += 1
        elif classification == "mismatch":
            mismatches.append(
                {
                    **base_loc,
                    "divisions": str(rec["divisions"]),
                    "duration_raw": rec["duration_raw"],
                    "actual_q": str(rec["actual_q"]),
                    "type": rec["type"],
                    "dots": rec["dots"],
                    "time_modification": rec["time_modification"],
                    "notated_q": str(notated_q),
                    "reason": reason,
                }
            )
        else:
            unsupported_items.append(
                {
                    **base_loc,
                    "type": rec["type"],
                    "dots": rec["dots"],
                    "time_modification": rec["time_modification"],
                    "actual_q": str(rec["actual_q"]) if rec["actual_q"] is not None else None,
                    "reason": reason,
                }
            )

    if mismatches:
        status = "fail"
    elif unsupported_items:
        status = "unsupported"
    else:
        status = "pass"

    return {
        "status": status,
        "input_id": input_id,
        "checked_items": len(parsed["notes"]),
        "consistent_count": consistent_count,
        "whole_measure_rest_count": whole_measure_rest_count,
        "mismatch_count": len(mismatches),
        "unsupported_count": len(unsupported_items),
        "mismatches": mismatches,
        "unsupported_items": unsupported_items,
    }


def extract_xml_strikes(xml_source: str | bytes | Path, input_id: str = "inline") -> dict[str, Any]:
    """Extract sounding XML strikes after resolving valid tie continuations."""
    parsed = _parse_musicxml(xml_source, input_id)
    if parsed["status"] != "ok":
        return {
            "status": "unsupported",
            "input_id": input_id,
            "reason": parsed["reason"],
            "unsupported_items": [{"reason": parsed["reason"]}],
        }

    blockers = list(parsed["blockers"])
    tie_anomalies: list[dict[str, Any]] = []
    events = sorted(parsed["pitched_events"], key=lambda e: (e["abs_q"], e["order"]))

    active: dict[tuple[str, str, F], dict[str, Any]] = {}
    strikes: list[dict[str, Any]] = []
    continuations: list[dict[str, Any]] = []

    for ev in events:
        if ev["grace"] or ev["actual_q"] is None:
            continue
        if ev["tieds"] - {"start", "stop"}:
            tie_anomalies.append(
                {
                    "code": "unsupported_tie_notation",
                    "part": ev["part"],
                    "measure": ev["measure"],
                    "abs_q": str(ev["abs_q"]),
                    "pitch": ev["pitch"],
                }
            )
        if ev["tieds"] and ev["tieds"] != ev["ties"]:
            tie_anomalies.append(
                {
                    "code": "tie_sound_notation_mismatch",
                    "part": ev["part"],
                    "measure": ev["measure"],
                    "abs_q": str(ev["abs_q"]),
                    "pitch": ev["pitch"],
                }
            )

        assert ev["midi"] is not None
        if ev["midi"].denominator != 1:
            blockers.append({"code": "microtonal_pitch_unsupported", "part": ev["part"], "measure": ev["measure"]})
            continue

        key = (ev["part"], ev["voice"], ev["midi"])
        is_continuation = False
        if "stop" in ev["ties"]:
            prev = active.pop(key, None)
            if prev is None or prev["abs_q"] + prev["actual_q"] != ev["abs_q"]:
                tie_anomalies.append(
                    {
                        "code": "orphan_tie_stop",
                        "part": ev["part"],
                        "measure": ev["measure"],
                        "measure_offset_q": str(ev["measure_offset_q"]),
                        "abs_q": str(ev["abs_q"]),
                        "midi": int(ev["midi"]),
                        "pitch": ev["pitch"],
                    }
                )
            else:
                is_continuation = True
        elif key in active:
            tie_anomalies.append(
                {
                    "code": "broken_tie_chain",
                    "part": ev["part"],
                    "measure": ev["measure"],
                    "measure_offset_q": str(ev["measure_offset_q"]),
                    "abs_q": str(ev["abs_q"]),
                    "midi": int(ev["midi"]),
                    "pitch": ev["pitch"],
                }
            )
            active.pop(key)

        record = {
            "part": ev["part"],
            "part_index": ev["part_index"],
            "measure": ev["measure"],
            "measure_index": ev["measure_index"],
            "measure_offset_q": ev["measure_offset_q"],
            "abs_q": ev["abs_q"],
            "dur_q": ev["actual_q"],
            "midi": int(ev["midi"]),
            "pitch": ev["pitch"],
            "voice": ev["voice"],
            "staff": ev["staff"],
        }
        if is_continuation:
            continuations.append(record)
        else:
            strikes.append(record)

        if "start" in ev["ties"]:
            active[key] = ev

    for unclosed in active.values():
        tie_anomalies.append(
            {
                "code": "unclosed_tie",
                "part": unclosed["part"],
                "measure": unclosed["measure"],
                "measure_offset_q": str(unclosed["measure_offset_q"]),
                "abs_q": str(unclosed["abs_q"]),
                "midi": int(unclosed["midi"]),
                "pitch": unclosed["pitch"],
            }
        )

    grouped: dict[tuple[int, F], list[dict[str, Any]]] = defaultdict(list)
    for st in strikes:
        grouped[(st["midi"], st["abs_q"])].append(st)

    unique_strikes: list[dict[str, Any]] = []
    duplicate_strikes: list[dict[str, Any]] = []
    for (midi_val, abs_q), grp in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        unique_strikes.append(grp[0])
        if len(grp) > 1:
            duplicate_strikes.append(
                {
                    "midi": midi_val,
                    "pitch": pitch_name(midi_val),
                    "abs_q": str(abs_q),
                    "count": len(grp),
                    "events": serialize_rationals(grp),
                }
            )

    if blockers or tie_anomalies:
        return {
            "status": "unsupported",
            "input_id": input_id,
            "reason": "xml_contains_blocking_constructs_or_tie_anomalies",
            "unsupported_items": serialize_rationals(blockers + tie_anomalies),
            "noteheads": len(events),
            "strike_noteheads": len(strikes),
            "unique_strike_count": len(unique_strikes),
            "measure_timeline": serialize_rationals(parsed["measure_timeline"]),
        }

    return {
        "status": "ok",
        "input_id": input_id,
        "noteheads": len(events),
        "strike_noteheads": len(strikes),
        "unique_strike_count": len(unique_strikes),
        "strikes": unique_strikes,
        "continuations": continuations,
        "duplicate_strikes": duplicate_strikes,
        "tie_anomalies": [],
        "measure_timeline": parsed["measure_timeline"],
        "total_q": parsed["total_q"],
    }


def _read_vlq(data: bytes, pos: int, end: int) -> tuple[int, int]:
    val = 0
    for _ in range(4):
        if pos >= end:
            raise ValueError("Truncated VLQ in MIDI track")
        b = data[pos]
        pos += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, pos
    raise ValueError("VLQ exceeds 4 bytes in MIDI track")


def parse_midi_note_ons(midi_source: bytes | Path | str, input_id: str = "inline") -> dict[str, Any]:
    """Parse SMF MIDI directly, converting PPQ ticks to exact rational quarter units."""
    try:
        data = _read_midi_bytes(midi_source)
    except OSError as exc:
        return {"status": "unsupported", "input_id": input_id, "reason": f"midi_read_error: {exc}"}

    if len(data) < 14 or data[:4] != b"MThd":
        return {"status": "unsupported", "input_id": input_id, "reason": "invalid_midi_header"}

    hdr_len = struct.unpack(">I", data[4:8])[0]
    if hdr_len < 6 or len(data) < 8 + hdr_len:
        return {"status": "unsupported", "input_id": input_id, "reason": "truncated_midi_header"}

    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])
    if (division & 0x8000) != 0:
        return {
            "status": "unsupported",
            "input_id": input_id,
            "reason": "unsupported_midi_timebase_smpte",
            "division_raw": hex(division),
        }
    if division == 0:
        return {"status": "unsupported", "input_id": input_id, "reason": "invalid_zero_ppq"}
    if fmt not in (0, 1):
        return {
            "status": "unsupported",
            "input_id": input_id,
            "reason": f"unsupported_midi_format_{fmt}",
        }

    ppq = division
    pos = 8 + hdr_len
    parsed_tracks = 0
    note_ons: list[dict[str, Any]] = []
    zero_vel_ignored = 0
    note_off_count = 0
    tempo_events: list[dict[str, Any]] = []
    invalid_tempo_events: list[dict[str, Any]] = []

    try:
        while parsed_tracks < ntrks:
            if pos + 8 > len(data):
                raise ValueError("Unexpected EOF before MTrk chunk")
            chunk_tag = data[pos : pos + 4]
            chunk_len = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
            pos += 8
            chunk_end = pos + chunk_len
            if chunk_end > len(data):
                raise ValueError("Truncated MIDI chunk")
            if chunk_tag != b"MTrk":
                pos = chunk_end
                continue

            track_idx = parsed_tracks
            parsed_tracks += 1
            tick = 0
            running_status: int | None = None

            while pos < chunk_end:
                delta, pos = _read_vlq(data, pos, chunk_end)
                tick += delta
                if pos >= chunk_end:
                    raise ValueError("Unexpected end of MTrk after delta")

                byte = data[pos]
                if byte < 0x80:
                    if running_status is None:
                        raise ValueError("Running status used before channel status byte")
                    status = running_status
                else:
                    pos += 1
                    status = byte
                    if status < 0xF0:
                        running_status = status
                    else:
                        running_status = None

                if status == 0xFF:
                    if pos >= chunk_end:
                        raise ValueError("Truncated meta event")
                    meta_type = data[pos]
                    pos += 1
                    meta_len, pos = _read_vlq(data, pos, chunk_end)
                    if pos + meta_len > chunk_end:
                        raise ValueError("Truncated meta event payload")
                    payload = data[pos : pos + meta_len]
                    pos += meta_len
                    if meta_type == 0x51:
                        if meta_len == 3:
                            us_per_q = int.from_bytes(payload, "big")
                            if us_per_q > 0:
                                tempo_events.append(
                                    {
                                        "track": track_idx,
                                        "tick": tick,
                                        "abs_q": F(tick, ppq),
                                        "us_per_quarter": us_per_q,
                                    }
                                )
                            else:
                                invalid_tempo_events.append(
                                    {
                                        "track": track_idx,
                                        "tick": tick,
                                        "abs_q": F(tick, ppq),
                                        "reason": "non_positive_us_per_quarter",
                                        "us_per_quarter": us_per_q,
                                    }
                                )
                        else:
                            invalid_tempo_events.append(
                                {
                                    "track": track_idx,
                                    "tick": tick,
                                    "abs_q": F(tick, ppq),
                                    "reason": f"invalid_tempo_meta_length_{meta_len}",
                                }
                            )
                elif status in (0xF0, 0xF7):
                    sysex_len, pos = _read_vlq(data, pos, chunk_end)
                    if pos + sysex_len > chunk_end:
                        raise ValueError("Truncated SysEx event")
                    pos += sysex_len
                elif status >= 0xF0:
                    data_bytes = 2 if status == 0xF2 else (1 if status in (0xF1, 0xF3) else 0)
                    if pos + data_bytes > chunk_end:
                        raise ValueError("Truncated system common message")
                    pos += data_bytes
                else:
                    msg_type = status & 0xF0
                    channel = status & 0x0F
                    if msg_type in (0xC0, 0xD0):
                        if pos + 1 > chunk_end:
                            raise ValueError("Truncated 1-byte channel voice message")
                        pos += 1
                    else:
                        if pos + 2 > chunk_end:
                            raise ValueError("Truncated 2-byte channel voice message")
                        d1 = data[pos]
                        d2 = data[pos + 1]
                        pos += 2
                        if msg_type == 0x80:
                            note_off_count += 1
                        elif msg_type == 0x90:
                            if d2 == 0:
                                zero_vel_ignored += 1
                            else:
                                note_ons.append(
                                    {
                                        "track": track_idx,
                                        "channel": channel,
                                        "tick": tick,
                                        "abs_q": F(tick, ppq),
                                        "midi": d1,
                                        "pitch": pitch_name(d1),
                                        "velocity": d2,
                                    }
                                )
            pos = chunk_end
    except ValueError as exc:
        return {"status": "unsupported", "input_id": input_id, "reason": f"midi_parse_error: {exc}"}

    note_ons.sort(key=lambda e: (e["abs_q"], e["midi"], e["track"], e["channel"]))
    tempo_events.sort(key=lambda e: (e["abs_q"], e["track"], e["tick"]))
    return {
        "status": "ok",
        "input_id": input_id,
        "format": fmt,
        "num_tracks": ntrks,
        "ppq": ppq,
        "real_note_on_count": len(note_ons),
        "zero_velocity_note_ons_ignored": zero_vel_ignored,
        "note_off_count": note_off_count,
        "tempo_events": tempo_events,
        "invalid_tempo_events": invalid_tempo_events,
        "note_ons": note_ons,
    }


def inspect_midi_constraints(midi_source: bytes | Path | str, input_id: str = "inline") -> dict[str, Any]:
    """Check raw SMF note-ons and tempo events against 30-note musicbox pitch, tempo, and gap constraints."""
    try:
        parsed = parse_midi_note_ons(midi_source, input_id=input_id)
    except OSError as exc:
        return {
            "status": "unsupported",
            "ok": False,
            "input_id": input_id,
            "reason": f"midi_read_error: {exc}",
            "tempo_verified": False,
            "tempo_check_scope": "unparsed: failed to read MIDI file",
        }

    if parsed.get("status") != "ok":
        return {
            "status": parsed.get("status", "unsupported"),
            "ok": False,
            "input_id": input_id,
            "reason": parsed.get("reason", "midi_parse_unsupported"),
            "tempo_verified": False,
            "tempo_check_scope": "unparsed: unsupported or corrupted MIDI structure",
        }

    note_ons: list[dict[str, Any]] = parsed["note_ons"]
    pitch_violations: list[dict[str, Any]] = [
        {
            "track": e["track"],
            "channel": e["channel"],
            "tick": e["tick"],
            "abs_q": e["abs_q"],
            "midi": e["midi"],
            "pitch": e["pitch"],
            "velocity": e["velocity"],
        }
        for e in note_ons
        if e["midi"] not in ALLOWED_MIDI
    ]

    strike_groups: dict[tuple[int, F], list[dict[str, Any]]] = defaultdict(list)
    for e in note_ons:
        strike_groups[(e["midi"], e["abs_q"])].append(e)

    unique_strikes: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for (midi, abs_q), group in strike_groups.items():
        unique_strikes.append(group[0])
        if len(group) > 1:
            duplicate_groups.append(
                {
                    "midi": midi,
                    "pitch": pitch_name(midi),
                    "abs_q": abs_q,
                    "count": len(group),
                    "strikes": [
                        {
                            "track": g["track"],
                            "channel": g["channel"],
                            "tick": g["tick"],
                            "abs_q": g["abs_q"],
                            "pitch": g["pitch"],
                            "velocity": g["velocity"],
                        }
                        for g in group
                    ],
                }
            )
    unique_strikes.sort(key=lambda e: (e["abs_q"], e["midi"], e["track"], e["channel"]))
    duplicate_groups.sort(key=lambda g: (g["abs_q"], g["midi"]))

    previous_by_midi: dict[int, dict[str, Any]] = {}
    gaps: list[F] = []
    too_close_pairs: list[dict[str, Any]] = []
    for e in unique_strikes:
        midi = e["midi"]
        prev = previous_by_midi.get(midi)
        if prev is not None:
            gap = e["abs_q"] - prev["abs_q"]
            gaps.append(gap)
            if gap < 1:
                too_close_pairs.append(
                    {
                        "gap_q": gap,
                        "midi": midi,
                        "pitch": e["pitch"],
                        "previous": {
                            "track": prev["track"],
                            "channel": prev["channel"],
                            "tick": prev["tick"],
                            "abs_q": prev["abs_q"],
                            "pitch": prev["pitch"],
                        },
                        "current": {
                            "track": e["track"],
                            "channel": e["channel"],
                            "tick": e["tick"],
                            "abs_q": e["abs_q"],
                            "pitch": e["pitch"],
                        },
                    }
                )
        previous_by_midi[midi] = e

    tempo_records = [
        {
            "track": t["track"],
            "tick": t["tick"],
            "abs_q": t["abs_q"],
            "us_per_quarter": t["us_per_quarter"],
            "quarter_bpm": F(60_000_000, t["us_per_quarter"]),
        }
        for t in parsed["tempo_events"]
    ]
    tempo_groups: dict[F, set[F]] = defaultdict(set)
    for t in tempo_records:
        tempo_groups[t["abs_q"]].add(t["quarter_bpm"])

    tempo_violations = [
        dict(t) for t in tempo_records if not (F(100) <= t["quarter_bpm"] <= F(220))
    ]
    tempo_conflicts = [
        {"abs_q": abs_q, "quarter_bpms": sorted(vals)}
        for abs_q, vals in sorted(tempo_groups.items())
        if len(vals) > 1
    ]
    invalid_tempo_events = parsed.get("invalid_tempo_events", [])
    missing_initial_tempo = (not tempo_records) or (not any(q <= 0 for q in tempo_groups))

    if invalid_tempo_events:
        tempo_verified = False
        tempo_check_scope = "unverified: malformed Set Tempo (0x51) meta event encountered"
    elif not tempo_records:
        tempo_verified = False
        tempo_check_scope = "unverified: no explicit Set Tempo (0x51) meta events in MIDI"
    elif missing_initial_tempo:
        tempo_verified = False
        tempo_check_scope = "unverified: explicit Set Tempo events appear only after q=0; initial tempo missing"
    elif tempo_conflicts:
        tempo_verified = False
        tempo_check_scope = "unverified: conflicting Set Tempo values at identical onset q"
    else:
        tempo_verified = True
        tempo_check_scope = "verified_explicit_tempo_timeline"

    empty_score = len(note_ons) == 0
    errors: list[str] = []
    if empty_score:
        errors.append("empty_score")
    if pitch_violations:
        errors.append("pitch_not_allowed")
    if too_close_pairs:
        errors.append("same_pitch_too_close")
    if duplicate_groups:
        errors.append("duplicate_strike")
    if missing_initial_tempo:
        errors.append("missing_initial_tempo")
    if tempo_conflicts:
        errors.append("tempo_conflict")
    if tempo_violations:
        errors.append("tempo_out_of_range")
    if invalid_tempo_events:
        errors.append("invalid_tempo_meta_event")

    pitch_range = (
        [pitch_name(min(e["midi"] for e in note_ons)), pitch_name(max(e["midi"] for e in note_ons))]
        if note_ons
        else None
    )
    ok = (not invalid_tempo_events) and tempo_verified and (len(errors) == 0)
    status = "unsupported" if invalid_tempo_events else ("pass" if ok else "fail")

    return {
        "status": status,
        "ok": ok,
        "input_id": input_id,
        "format": parsed["format"],
        "num_tracks": parsed["num_tracks"],
        "ppq": parsed["ppq"],
        "real_note_on_count": len(note_ons),
        "unique_strike_count": len(unique_strikes),
        "zero_velocity_note_ons_ignored": parsed["zero_velocity_note_ons_ignored"],
        "note_off_count": parsed["note_off_count"],
        "pitch_range": pitch_range,
        "pitch_violation_count": len(pitch_violations),
        "pitch_violations": pitch_violations,
        "minimum_same_pitch_gap_q": min(gaps) if gaps else None,
        "too_close_pair_count": len(too_close_pairs),
        "too_close_pairs": too_close_pairs,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "tempos": tempo_records,
        "tempo_violation_count": len(tempo_violations),
        "tempo_violations": tempo_violations,
        "tempo_conflict_count": len(tempo_conflicts),
        "tempo_conflicts": tempo_conflicts,
        "invalid_tempo_events": invalid_tempo_events,
        "missing_initial_tempo": missing_initial_tempo,
        "tempo_verified": tempo_verified,
        "tempo_check_scope": tempo_check_scope,
        "errors": errors,
    }


def _locate_abs_q_in_timeline(abs_q: F, timeline: list[dict[str, Any]]) -> tuple[str | None, F | None]:
    for bar in timeline:
        if bar["start_q"] <= abs_q < bar["end_q"]:
            return bar["measure"], abs_q - bar["start_q"]
    if timeline and abs_q == timeline[-1]["end_q"]:
        return timeline[-1]["measure"], abs_q - timeline[-1]["start_q"]
    return None, None


def compare_xml_and_midi(
    xml_source: str | bytes | Path,
    midi_source: bytes | Path | str,
    xml_id: str = "inline_xml",
    midi_id: str = "inline_midi",
) -> dict[str, Any]:
    """Compare XML tie-resolved strikes against real MIDI note-ons by (midi, abs_q)."""
    xml_res = extract_xml_strikes(xml_source, input_id=xml_id)
    midi_res = parse_midi_note_ons(midi_source, input_id=midi_id)

    unsupported_reasons: list[str] = []
    if xml_res["status"] != "ok":
        unsupported_reasons.append(f"xml:{xml_res.get('reason', 'unsupported')}")
    if midi_res["status"] != "ok":
        unsupported_reasons.append(f"midi:{midi_res.get('reason', 'unsupported')}")

    if unsupported_reasons:
        return {
            "status": "unsupported",
            "xml_id": xml_id,
            "midi_id": midi_id,
            "xml_noteheads": xml_res.get("noteheads"),
            "xml_strike_noteheads": xml_res.get("strike_noteheads"),
            "xml_unique_strikes": xml_res.get("unique_strike_count"),
            "midi_real_note_ons": midi_res.get("real_note_on_count"),
            "midi_zero_velocity_note_ons_ignored": midi_res.get("zero_velocity_note_ons_ignored"),
            "midi_ppq": midi_res.get("ppq"),
            "matched_count": None,
            "extra_in_midi_count": None,
            "missing_in_midi_count": None,
            "extra_in_midi": [],
            "missing_in_midi": [],
            "unsupported_reasons": unsupported_reasons,
            "xml_unsupported_items": xml_res.get("unsupported_items", []),
        }

    xml_by_key: dict[tuple[int, F], list[dict[str, Any]]] = defaultdict(list)
    for st in xml_res["strikes"]:
        xml_by_key[(st["midi"], st["abs_q"])].append(st)

    midi_by_key: dict[tuple[int, F], list[dict[str, Any]]] = defaultdict(list)
    for ev in midi_res["note_ons"]:
        midi_by_key[(ev["midi"], ev["abs_q"])].append(ev)

    cont_keys = {(c["midi"], c["abs_q"]): c for c in xml_res["continuations"]}
    timeline = xml_res["measure_timeline"]

    all_keys = sorted(set(xml_by_key) | set(midi_by_key), key=lambda k: (k[1], k[0]))
    matched_count = 0
    extra_in_midi: list[dict[str, Any]] = []
    missing_in_midi: list[dict[str, Any]] = []

    for key in all_keys:
        midi_val, abs_q = key
        x_list = xml_by_key.get(key, [])
        m_list = midi_by_key.get(key, [])
        common = min(len(x_list), len(m_list))
        matched_count += common

        if len(m_list) > len(x_list):
            measure_num, measure_off = _locate_abs_q_in_timeline(abs_q, timeline)
            cont = cont_keys.get(key)
            for m_ev in m_list[len(x_list) :]:
                extra_in_midi.append(
                    {
                        "midi": midi_val,
                        "pitch": pitch_name(midi_val),
                        "abs_q": str(abs_q),
                        "measure": cont["measure"] if cont else measure_num,
                        "measure_offset_q": str(cont["measure_offset_q"] if cont else measure_off)
                        if (cont or measure_off is not None)
                        else None,
                        "part": cont["part"] if cont else None,
                        "tick": m_ev["tick"],
                        "track": m_ev["track"],
                        "channel": m_ev["channel"],
                        "velocity": m_ev["velocity"],
                        "matches_xml_tie_continuation": cont is not None,
                        "reason": (
                            "extra_note_on_at_xml_tie_continuation"
                            if cont is not None
                            else "extra_note_on_not_in_xml_strikes"
                        ),
                    }
                )

        if len(x_list) > len(m_list):
            for x_ev in x_list[len(m_list) :]:
                missing_in_midi.append(
                    {
                        "midi": midi_val,
                        "pitch": pitch_name(midi_val),
                        "abs_q": str(abs_q),
                        "part": x_ev["part"],
                        "measure": x_ev["measure"],
                        "measure_offset_q": str(x_ev["measure_offset_q"]),
                        "reason": "xml_strike_missing_in_midi",
                    }
                )

    has_xml_duplicates = bool(xml_res["duplicate_strikes"])
    status = "pass" if (not extra_in_midi and not missing_in_midi and not has_xml_duplicates) else "fail"
    return {
        "status": status,
        "xml_id": xml_id,
        "midi_id": midi_id,
        "xml_noteheads": xml_res["noteheads"],
        "xml_strike_noteheads": xml_res["strike_noteheads"],
        "xml_unique_strikes": xml_res["unique_strike_count"],
        "xml_duplicate_strikes_count": len(xml_res["duplicate_strikes"]),
        "xml_duplicate_strikes": xml_res["duplicate_strikes"],
        "midi_real_note_ons": midi_res["real_note_on_count"],
        "midi_zero_velocity_note_ons_ignored": midi_res["zero_velocity_note_ons_ignored"],
        "midi_ppq": midi_res["ppq"],
        "matched_count": matched_count,
        "extra_in_midi_count": len(extra_in_midi),
        "missing_in_midi_count": len(missing_in_midi),
        "extra_in_midi": extra_in_midi,
        "missing_in_midi": missing_in_midi,
        "unsupported_reasons": [],
    }


def summarize_xml_transposition(xml_source: str | bytes | Path, input_id: str = "inline") -> dict[str, Any]:
    """Return written vs sounding pitch ranges and declared <transpose> records."""
    parsed = _parse_musicxml(xml_source, input_id)
    if parsed["status"] != "ok":
        return {"status": "unsupported", "input_id": input_id, "reason": parsed["reason"]}
    written = [ev["written_midi"] for ev in parsed["pitched_events"] if ev["written_midi"] is not None]
    sounding = [ev["midi"] for ev in parsed["pitched_events"] if ev["midi"] is not None]
    return {
        "status": "ok",
        "input_id": input_id,
        "transpositions": serialize_rationals(parsed["transpose_events"]),
        "written_pitch_range": [pitch_name(min(written)), pitch_name(max(written))] if written else [],
        "sounding_pitch_range": [pitch_name(min(sounding)), pitch_name(max(sounding))] if sounding else [],
    }


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    """Return True when two paths resolve to the same target path or file identity."""
    first_abs = os.path.normcase(os.path.abspath(os.fspath(first)))
    second_abs = os.path.normcase(os.path.abspath(os.fspath(second)))
    if first_abs == second_abs:
        return True
    try:
        first_res = os.path.normcase(os.fspath(first.resolve(strict=False)))
        second_res = os.path.normcase(os.fspath(second.resolve(strict=False)))
        if first_res == second_res:
            return True
    except OSError:
        pass
    try:
        if first.exists() and second.exists() and os.path.samefile(first, second):
            return True
    except OSError:
        pass
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MusicXML graphic duration and MIDI onset parity.")
    parser.add_argument("--xml", type=Path, required=True, help="Path to MusicXML (.musicxml/.xml/.mxl)")
    parser.add_argument("--midi", type=Path, help="Optional path to SMF MIDI (.mid)")
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    if args.json is not None:
        for flag_name, input_path in (("--xml", args.xml), ("--midi", args.midi)):
            if input_path is not None and _paths_refer_to_same_file(args.json, input_path):
                print(
                    f"Refusing to write --json output '{args.json}': collides with {flag_name} input '{input_path}'.",
                    file=sys.stderr,
                )
                return 2

    out: dict[str, Any] = {
        "notation_durations": inspect_notation_durations(args.xml, input_id=args.xml.name),
    }
    if args.midi is not None:
        out["midi_onset_comparison"] = compare_xml_and_midi(
            args.xml, args.midi, xml_id=args.xml.name, midi_id=args.midi.name
        )

    payload = json.dumps(serialize_rationals(out), ensure_ascii=False, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
