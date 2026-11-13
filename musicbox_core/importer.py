"""MusicXML linear partwise importer built over verified project parsers."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction as F
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from musicbox_core.model import (
    MeasureSpan,
    NoteEvent,
    ScoreIR,
    SourceRef,
    TempoEvent,
    UnsupportedScoreError,
)

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from inspect_notation_midi import inspect_notation_durations  # noqa: E402

LEGACY_AUDITOR_PATH = TOOLS_DIR / "audit_musicxml.py"

# Target instrument rules do not constrain input piano scores; structural blockers DO.
STRUCTURAL_BLOCKER_CODES = frozenset(
    {
        "unsupported_score_type",
        "unsupported_meter",
        "unsupported_transposition_doubling",
        "unresolved_midmeasure_transposition",
        "invalid_cursor_duration",
        "negative_cursor",
        "unresolved_performance_timing",
        "missing_divisions",
        "invalid_note_duration",
        "unresolved_grace",
        "orphan_chord",
        "unsupported_tie_notation",
        "tie_sound_notation_mismatch",
        "conditional_tie",
        "unpitched_or_unknown_note",
        "unresolved_ornament",
        "unexpanded_navigation",
        "conditional_sound",
        "unresolved_metronome",
        "unresolved_octave_shift",
        "unresolved_playback_text",
        "overfull_measure",
        "underfull_measure",
        "part_measure_count_mismatch",
        "part_measure_length_mismatch",
        "part_measure_number_mismatch",
    }
)


def _load_legacy_auditor():
    """Dynamically load legacy audit_musicxml without writing __pycache__."""
    mod_name = "_musicbox_legacy_audit_musicxml"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    prev_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(mod_name, LEGACY_AUDITOR_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load legacy auditor from {LEGACY_AUDITOR_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = prev_dont_write


def _inspect_aux_xml_structure(
    raw_bytes: bytes,
) -> tuple[dict[str, str], dict[tuple[str, int, int], int], list[dict[str, Any]]]:
    """Collect part names, per-measure 1-based <note> indices, and subtle meter checks."""
    root = ET.fromstring(raw_bytes)
    for elem in root.iter():
        elem.tag = elem.tag.split("}")[-1]

    part_names: dict[str, str] = {}
    for idx, sp in enumerate(root.findall("./part-list/score-part")):
        pid = sp.get("id", str(idx + 1))
        name = (sp.findtext("part-name") or pid).strip() or pid
        part_names[pid] = name

    note_index_by_key: dict[tuple[str, int, int], int] = {}
    extra_blockers: list[dict[str, Any]] = []
    pitched_counter = 0

    for pi, part in enumerate(root.findall("part")):
        pid = part.get("id", str(pi + 1))
        part_names.setdefault(pid, pid)
        for mi, measure in enumerate(part.findall("measure")):
            measure_num = measure.get("number", str(mi + 1))
            note_idx = 0
            for item in measure:
                if item.tag == "attributes":
                    time_elem = item.find("time")
                    if time_elem is not None:
                        if time_elem.find("senza-misura") is not None or time_elem.find("interchangeable") is not None:
                            extra_blockers.append(
                                {
                                    "code": "unsupported_meter",
                                    "part": pid,
                                    "measure": measure_num,
                                    "measure_index": mi,
                                    "detail": "senza-misura or interchangeable meter is unsupported",
                                }
                            )
                        beats = time_elem.findall("beats")
                        types = time_elem.findall("beat-type")
                        if len(beats) == 1 and len(types) == 1:
                            b_txt = (beats[0].text or "").strip()
                            t_txt = (types[0].text or "").strip()
                            if "+" in b_txt:
                                extra_blockers.append(
                                    {
                                        "code": "unsupported_meter",
                                        "part": pid,
                                        "measure": measure_num,
                                        "measure_index": mi,
                                        "detail": f"additive meter {b_txt}/{t_txt} is unsupported",
                                    }
                                )
                            else:
                                try:
                                    b_val = int(b_txt)
                                    t_val = int(t_txt)
                                except ValueError:
                                    extra_blockers.append(
                                        {
                                            "code": "unsupported_meter",
                                            "part": pid,
                                            "measure": measure_num,
                                            "measure_index": mi,
                                            "detail": f"non-integer meter {b_txt}/{t_txt}",
                                        }
                                    )
                                    b_val, t_val = 0, 0
                                if b_val <= 0 or t_val <= 0 or (t_val & (t_val - 1)) != 0:
                                    extra_blockers.append(
                                        {
                                            "code": "unsupported_meter",
                                            "part": pid,
                                            "measure": measure_num,
                                            "measure_index": mi,
                                            "detail": f"irregular beat-type {b_txt}/{t_txt}",
                                        }
                                    )
                        elif len(beats) > 1 or len(types) > 1:
                            extra_blockers.append(
                                {
                                    "code": "unsupported_meter",
                                    "part": pid,
                                    "measure": measure_num,
                                    "measure_index": mi,
                                    "detail": "compound dual meter element is unsupported",
                                }
                            )
                elif item.tag == "note":
                    note_idx += 1
                    if item.find("pitch") is not None:
                        note_index_by_key[(pid, mi, pitched_counter)] = note_idx
                        pitched_counter += 1
                    elif item.find("grace") is not None:
                        pass

    return part_names, note_index_by_key, extra_blockers


def load_score(path: Path | str) -> ScoreIR:
    """Parse a MusicXML (.musicxml/.xml/.mxl) file into an exact-rational ScoreIR.

    Raises UnsupportedScoreError with part/measure locations on unsupported structures.
    """
    score_path = Path(path)
    if not score_path.is_file():
        raise UnsupportedScoreError(
            f"Input file not found: {score_path}",
            issues=[{"code": "file_not_found", "path": str(score_path)}],
        )

    am = _load_legacy_auditor()
    try:
        file_bytes = score_path.read_bytes()
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        raw_xml = am.read_xml(score_path)
        parsed = am.parse_text(raw_xml)
        part_names, note_index_by_key, extra_blockers = _inspect_aux_xml_structure(raw_xml)
    except (ET.ParseError, ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        raise UnsupportedScoreError(
            f"Failed to parse MusicXML '{score_path}': {exc}",
            issues=[{"code": "xml_parse_error", "path": str(score_path), "detail": str(exc)}],
        ) from exc

    blockers: list[dict[str, Any]] = []
    for err in parsed["errors"]:
        if err.get("code") in STRUCTURAL_BLOCKER_CODES:
            blockers.append(dict(err))
    blockers.extend(extra_blockers)

    # Check fractional pitches (microtones) and build exact tie-merged events.
    events_sorted = sorted(parsed["events"], key=lambda e: (e["onset"], e["order"]))
    active: dict[tuple[str, str, int], dict[str, Any]] = {}
    part_strikes: list[dict[str, Any]] = []

    for ev in events_sorted:
        midi_frac = F(ev["midi"])
        if midi_frac.denominator != 1:
            blockers.append(
                {
                    "code": "microtonal_pitch_unsupported",
                    "part": ev["part"],
                    "measure": ev["measure"],
                    "measure_index": ev["measure_index"],
                    "offset": ev["offset"],
                    "midi": str(midi_frac),
                }
            )
            continue
        midi_int = int(midi_frac)
        note_idx = note_index_by_key.get((ev["part"], ev["measure_index"], ev["order"]), ev["order"] + 1)
        ref = SourceRef(
            part=ev["part"],
            measure=str(ev["measure"]),
            measure_index=int(ev["measure_index"]),
            note_index=note_idx,
            onset_q=F(ev["onset"]),
            duration_q=F(ev["dur"]),
            voice=str(ev["voice"]),
            staff=str(ev["staff"]),
            midi=midi_int,
            pitch=ev["pitch"],
        )

        key = (ev["part"], str(ev["voice"]), midi_int)
        if "stop" in ev["ties"]:
            prev = active.pop(key, None)
            if prev is None or prev["end_q"] != F(ev["onset"]):
                blockers.append(
                    {
                        "code": "orphan_tie_stop",
                        "part": ev["part"],
                        "measure": ev["measure"],
                        "measure_index": ev["measure_index"],
                        "offset": ev["offset"],
                        "pitch": ev["pitch"],
                    }
                )
                continue
            prev["duration_q"] += F(ev["dur"])
            prev["end_q"] = prev["onset_q"] + prev["duration_q"]
            prev["refs"].append(ref)
            chain = prev
        else:
            if key in active:
                old = active.pop(key)
                blockers.append(
                    {
                        "code": "broken_tie_chain",
                        "part": ev["part"],
                        "measure": ev["measure"],
                        "measure_index": ev["measure_index"],
                        "offset": ev["offset"],
                        "pitch": ev["pitch"],
                        "previous_measure": old["refs"][-1].measure,
                    }
                )
            chain = {
                "part": ev["part"],
                "part_index": int(ev["part_index"]),
                "voice": str(ev["voice"]),
                "staff": str(ev["staff"]),
                "midi": midi_int,
                "onset_q": F(ev["onset"]),
                "duration_q": F(ev["dur"]),
                "end_q": F(ev["onset"]) + F(ev["dur"]),
                "refs": [ref],
                "order": ev["order"],
            }
            part_strikes.append(chain)

        if "start" in ev["ties"]:
            active[key] = chain

    for unclosed in active.values():
        last_ref = unclosed["refs"][-1]
        blockers.append(
            {
                "code": "unclosed_tie",
                "part": last_ref.part,
                "measure": last_ref.measure,
                "measure_index": last_ref.measure_index,
                "offset": last_ref.onset_q,
                "pitch": last_ref.pitch,
            }
        )

    # Validate explicit tempo timeline (must start by q<=0, positive, non-conflicting).
    tempo_groups: dict[F, set[F]] = defaultdict(set)
    tempo_events: list[TempoEvent] = []
    seen_tempo_keys: set[tuple[F, F]] = set()
    for t in sorted(parsed["tempos"], key=lambda x: (x["onset"], x["qpm"])):
        onset_q = F(t["onset"])
        qpm = F(t["qpm"])
        if qpm <= 0:
            blockers.append(
                {
                    "code": "invalid_tempo",
                    "part": t.get("part"),
                    "measure": t.get("measure"),
                    "measure_index": t.get("measure_index"),
                    "onset": onset_q,
                    "qpm": qpm,
                }
            )
            continue
        tempo_groups[onset_q].add(qpm)
        if (onset_q, qpm) not in seen_tempo_keys:
            seen_tempo_keys.add((onset_q, qpm))
            tempo_events.append(TempoEvent(onset_q=onset_q, quarter_bpm=qpm, source=str(t.get("source", "score"))))

    if not tempo_events or not any(onset <= 0 for onset in tempo_groups):
        blockers.append({"code": "missing_initial_tempo", "detail": "No explicit initial tempo at or before onset 0"})
    for onset_q, values in sorted(tempo_groups.items()):
        if len(values) > 1:
            blockers.append(
                {
                    "code": "tempo_conflict",
                    "onset": onset_q,
                    "values": [str(v) for v in sorted(values)],
                }
            )

    if not parsed["parts"] or not part_strikes:
        blockers.append({"code": "empty_score", "detail": "No playable pitched notes found in score"})

    if blockers:
        first = blockers[0]
        loc_bits = []
        if first.get("part") is not None:
            loc_bits.append(f"part={first['part']}")
        if first.get("measure") is not None:
            loc_bits.append(f"measure={first['measure']}")
        loc_str = f" ({', '.join(loc_bits)})" if loc_bits else ""
        raise UnsupportedScoreError(
            f"Unsupported score structure: {first.get('code')}{loc_str}; total issues={len(blockers)}",
            issues=blockers,
        )

    # Build canonical measure spans using time signatures from attributes.
    time_sig_by_mi: dict[int, tuple[int, int]] = {}
    for attr in parsed["attributes"]:
        if attr.get("kind") == "time" and attr.get("value"):
            b_str, t_str = attr["value"][0]
            time_sig_by_mi.setdefault(int(attr["measure_index"]), (int(b_str), int(t_str)))

    ref_part = parsed["parts"][0]
    ref_measures = [m for m in parsed["measures"] if m["part"] == ref_part]
    measures: list[MeasureSpan] = []
    cur_beats, cur_beat_type = 4, 4
    for m in ref_measures:
        mi = int(m["measure_index"])
        if mi in time_sig_by_mi:
            cur_beats, cur_beat_type = time_sig_by_mi[mi]
        start_q = F(m["onset"])
        len_q = F(m["length"])
        measures.append(
            MeasureSpan(
                measure_index=mi,
                measure=str(m["measure"]),
                start_q=start_q,
                end_q=start_q + len_q,
                length_q=len_q,
                nominal_q=F(m["nominal"]),
                beats=cur_beats,
                beat_type=cur_beat_type,
                implicit=(m.get("implicit") == "yes"),
            )
        )

    # Deduplicate simultaneous strikes sharing (part, midi, onset_q) while keeping all SourceRefs.
    grouped_part_strikes: dict[tuple[str, int, F], list[dict[str, Any]]] = defaultdict(list)
    for st in part_strikes:
        grouped_part_strikes[(st["part"], st["midi"], st["onset_q"])].append(st)

    note_events: list[NoteEvent] = []
    for idx, ((part_id, midi_val, onset_q), grp) in enumerate(
        sorted(grouped_part_strikes.items(), key=lambda kv: (kv[0][2], kv[1][0]["part_index"], kv[0][1])),
        start=1,
    ):
        max_dur = max(item["duration_q"] for item in grp)
        merged_refs: list[SourceRef] = []
        for item in grp:
            merged_refs.extend(item["refs"])
        first = grp[0]
        note_events.append(
            NoteEvent(
                id=f"src_{idx:05d}",
                midi=midi_val,
                onset_q=onset_q,
                duration_q=max_dur,
                role="source",
                source_refs=merged_refs,
                part=part_id,
                voice=first["voice"],
                staff=first["staff"],
                original_midi=midi_val,
            )
        )

    # Graphic duration mismatches are non-blocking warnings for source import; record them explicitly.
    warnings: list[dict[str, Any]] = []
    notation_diag = inspect_notation_durations(raw_xml, input_id=score_path.name)
    if notation_diag.get("mismatch_count", 0) > 0:
        warnings.append(
            {
                "code": "graphic_notation_duration_mismatch",
                "mismatch_count": notation_diag["mismatch_count"],
                "sample": notation_diag["mismatches"][:5],
                "note": "Events built using exact duration/divisions values.",
            }
        )
    if notation_diag.get("unsupported_count", 0) > 0:
        warnings.append(
            {
                "code": "graphic_notation_unsupported_items",
                "unsupported_count": notation_diag["unsupported_count"],
                "sample": notation_diag["unsupported_items"][:5],
            }
        )

    transpositions_recorded = [
        dict(a) for a in parsed["attributes"] if a.get("kind") == "transpose"
    ]

    return ScoreIR(
        parts=list(parsed["parts"]),
        part_names=part_names,
        events=note_events,
        tempos=tempo_events,
        measures=measures,
        total_q=F(parsed["total_q"]),
        source_path=str(score_path),
        source_sha256=sha256,
        warnings=warnings,
        unsupported_issues=[],
        metadata={
            "source_transpositions": transpositions_recorded,
            "raw_pitched_notehead_count": len(parsed["events"]),
        },
    )
