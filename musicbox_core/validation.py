"""Three-layer output validation against MusicXML, raw SMF MIDI, ScoreIR, and music21."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import struct
import sys
from typing import Any

from music21 import chord as m21_chord
from music21 import converter as m21_converter
from music21 import note as m21_note

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from inspect_notation_midi import (  # noqa: E402
    _read_midi_bytes,
    _read_vlq,
    compare_xml_and_midi,
    inspect_midi_constraints,
    inspect_notation_durations,
    summarize_xml_transposition,
)

from .exporter import encode_tempo_us
from .importer import LEGACY_AUDITOR_PATH, load_score
from .model import ALLOWED_MIDI, ScoreIR, pitch_name, serialize_rationals


def _import_audit_module() -> Any:
    prev_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_musicbox_audit_validation", LEGACY_AUDITOR_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load audit script at {LEGACY_AUDITOR_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = prev_dont_write


def extract_midi_note_intervals(midi_path: Path | str) -> dict[str, Any]:
    """Parse raw SMF bytes from disk to reconstruct exact rational (midi, onset_q, duration_q) intervals."""
    p = Path(midi_path)
    data = _read_midi_bytes(p)
    if len(data) < 14 or data[:4] != b"MThd":
        return {"status": "unsupported", "reason": "invalid_midi_header"}
    hdr_len = struct.unpack(">I", data[4:8])[0]
    if hdr_len < 6 or len(data) < 8 + hdr_len:
        return {"status": "unsupported", "reason": "truncated_midi_header"}
    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])
    if (division & 0x8000) != 0 or division == 0 or fmt not in (0, 1):
        return {"status": "unsupported", "reason": f"unsupported_header_fmt_{fmt}_div_{division}"}

    ppq = division
    pos = 8 + hdr_len
    parsed_tracks = 0
    active: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    intervals: list[dict[str, Any]] = []
    orphan_note_offs = 0
    tempo_events: list[dict[str, Any]] = []
    track_eot_ticks: dict[int, int] = {}

    try:
        while parsed_tracks < ntrks:
            if pos + 8 > len(data):
                raise ValueError("Unexpected EOF before MTrk")
            tag = data[pos : pos + 4]
            clen = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
            pos += 8
            cend = pos + clen
            if cend > len(data):
                raise ValueError("Truncated MTrk")
            if tag != b"MTrk":
                pos = cend
                continue

            track_idx = parsed_tracks
            parsed_tracks += 1
            tick = 0
            running_status: int | None = None

            while pos < cend:
                delta, pos = _read_vlq(data, pos, cend)
                tick += delta
                if pos >= cend:
                    raise ValueError("Unexpected end of MTrk after delta")
                byte = data[pos]
                if byte < 0x80:
                    if running_status is None:
                        raise ValueError("Running status without prior status")
                    status = running_status
                else:
                    pos += 1
                    status = byte
                    running_status = status if status < 0xF0 else None

                if status == 0xFF:
                    if pos >= cend:
                        raise ValueError("Truncated meta event")
                    meta_type = data[pos]
                    pos += 1
                    mlen, pos = _read_vlq(data, pos, cend)
                    if pos + mlen > cend:
                        raise ValueError("Truncated meta payload")
                    payload = data[pos : pos + mlen]
                    pos += mlen
                    if meta_type == 0x51 and mlen == 3:
                        us_per_q = int.from_bytes(payload, "big")
                        tempo_events.append(
                            {
                                "track": track_idx,
                                "tick": tick,
                                "abs_q": F(tick, ppq),
                                "us_per_quarter": us_per_q,
                                "bpm": float(F(60_000_000, us_per_q)) if us_per_q > 0 else None,
                            }
                        )
                    elif meta_type == 0x2F:
                        track_eot_ticks[track_idx] = tick
                elif status in (0xF0, 0xF7):
                    slen, pos = _read_vlq(data, pos, cend)
                    if pos + slen > cend:
                        raise ValueError("Truncated SysEx")
                    pos += slen
                elif status >= 0xF0:
                    dbytes = 2 if status == 0xF2 else (1 if status in (0xF1, 0xF3) else 0)
                    pos += dbytes
                else:
                    mtype = status & 0xF0
                    ch = status & 0x0F
                    if mtype in (0xC0, 0xD0):
                        pos += 1
                    else:
                        if pos + 2 > cend:
                            raise ValueError("Truncated channel message")
                        d1 = data[pos]
                        d2 = data[pos + 1]
                        pos += 2
                        if mtype == 0x90 and d2 > 0:
                            active[(track_idx, ch, d1)].append((tick, d2))
                        elif mtype == 0x80 or (mtype == 0x90 and d2 == 0):
                            key = (track_idx, ch, d1)
                            if not active[key]:
                                orphan_note_offs += 1
                            else:
                                on_tick, vel = active[key].pop(0)
                                dur_tick = tick - on_tick
                                intervals.append(
                                    {
                                        "track": track_idx,
                                        "channel": ch,
                                        "midi": d1,
                                        "pitch": pitch_name(d1),
                                        "onset_q": F(on_tick, ppq),
                                        "duration_q": F(dur_tick, ppq),
                                        "velocity": vel,
                                    }
                                )
            pos = cend
    except ValueError as exc:
        return {"status": "unsupported", "reason": f"midi_parse_error: {exc}"}

    unclosed_notes = sum(len(v) for v in active.values())
    intervals.sort(key=lambda e: (e["onset_q"], e["midi"], e["track"]))
    tempo_events.sort(key=lambda e: (e["abs_q"], e["track"]))
    return {
        "status": "ok",
        "format": fmt,
        "num_tracks": ntrks,
        "ppq": ppq,
        "intervals": intervals,
        "unclosed_notes_count": unclosed_notes,
        "orphan_note_off_count": orphan_note_offs,
        "tempo_events": tempo_events,
        "track0_eot_q": F(track_eot_ticks[0], ppq) if 0 in track_eot_ticks else None,
        "max_eot_q": F(max(track_eot_ticks.values()), ppq) if track_eot_ticks else None,
    }


def _extract_m21_strike_counter(score_stream: Any) -> Counter[tuple[int, F]]:
    """Extract (midi, onset_q) strike multiset from a music21 stream, ignoring tie continuations."""
    counter: Counter[tuple[int, F]] = Counter()
    for el in score_stream.flatten().notes:
        off_q = F(el.offset).limit_denominator(4096)
        if isinstance(el, m21_note.Note):
            if el.tie is None or el.tie.type == "start":
                counter[(int(el.pitch.midi), off_q)] += 1
        elif isinstance(el, m21_chord.Chord):
            for sub_n in el.notes:
                tie_obj = sub_n.tie or el.tie
                if tie_obj is None or tie_obj.type == "start":
                    counter[(int(sub_n.pitch.midi), off_q)] += 1
    return counter


def _music21_cross_check(xml_path: Path, midi_path: Path, score: ScoreIR) -> dict[str, Any]:
    """Compare onset+pitch multisets across independent music21 XML and MIDI parses and ScoreIR."""
    m21_xml = m21_converter.parse(str(xml_path), forceSource=True)
    m21_midi = m21_converter.parse(str(midi_path), forceSource=True, quantizePost=False)

    xml_counter = _extract_m21_strike_counter(m21_xml)
    midi_counter = _extract_m21_strike_counter(m21_midi)
    ir_counter: Counter[tuple[int, F]] = Counter((e.midi, e.onset_q) for e in score.events)

    xml_eq_midi = xml_counter == midi_counter
    xml_eq_ir = xml_counter == ir_counter
    midi_eq_ir = midi_counter == ir_counter
    ok = xml_eq_midi and xml_eq_ir and midi_eq_ir
    return {
        "status": "pass" if ok else "fail",
        "xml_strike_count": sum(xml_counter.values()),
        "midi_strike_count": sum(midi_counter.values()),
        "ir_strike_count": sum(ir_counter.values()),
        "xml_equals_midi": xml_eq_midi,
        "xml_equals_ir": xml_eq_ir,
        "midi_equals_ir": midi_eq_ir,
    }


def _compute_duration_seconds(score: ScoreIR) -> float:
    """Integrate exact duration seconds along the ScoreIR tempo timeline."""
    tempos = sorted(score.tempos, key=lambda t: t.onset_q)
    if not tempos:
        return 0.0
    total_sec = F(0)
    cur_bpm = tempos[0].quarter_bpm
    prev_q = F(0)
    for t in tempos:
        if t.onset_q > prev_q:
            span = min(t.onset_q, score.total_q) - prev_q
            if span > 0:
                total_sec += span * F(60) / cur_bpm
            prev_q = t.onset_q
        cur_bpm = t.quarter_bpm
    if score.total_q > prev_q:
        total_sec += (score.total_q - prev_q) * F(60) / cur_bpm
    return round(float(total_sec), 3)


def validate_variant(score: ScoreIR, paths: dict[str, Path]) -> dict[str, Any]:
    """Validate exported MusicXML + MIDI artifacts on disk across Layers 1, 2, and 3."""
    xml_path = Path(paths["musicxml"])
    midi_path = Path(paths["midi"])
    failure_reasons: list[str] = []

    if not xml_path.is_file():
        failure_reasons.append(f"missing_musicxml:{xml_path}")
    if not midi_path.is_file():
        failure_reasons.append(f"missing_midi:{midi_path}")
    if failure_reasons:
        return {
            "status": "fail",
            "passed": False,
            "failure_reasons": failure_reasons,
            "layer1_constraints": {"status": "fail"},
            "layer2_fidelity": {"status": "fail"},
            "layer3_musical_summary": {"listening_evaluation": "not_evaluated"},
        }

    # --- Layer 1: Independent Physical Constraint Checks on Disk Artifacts ---
    audit_mod = _import_audit_module()
    xml_text = xml_path.read_text(encoding="utf-8", errors="replace")
    xml_audit = audit_mod.audit_text(xml_text)
    xml_l1_ok = bool(xml_audit.get("ok")) and len(xml_audit["errors"]) == 0
    if not xml_l1_ok:
        failure_reasons.append(f"layer1_xml_audit_errors:{xml_audit['errors']}")

    midi_constraints = inspect_midi_constraints(midi_path, input_id=midi_path.name)
    midi_l1_ok = midi_constraints.get("status") == "pass" and bool(midi_constraints.get("ok"))
    if not midi_l1_ok:
        failure_reasons.append(f"layer1_midi_constraints_failed:{midi_constraints.get('errors') or midi_constraints.get('reason')}")

    layer1_ok = xml_l1_ok and midi_l1_ok
    audit_qpm = [t["qpm"] for t in xml_audit.get("tempo_events", [])]
    layer1 = {
        "status": "pass" if layer1_ok else "fail",
        "xml_audit": {
            "status": "pass" if xml_l1_ok else "fail",
            "ok": xml_audit.get("ok"),
            "errors": xml_audit["errors"],
            "error_counts": xml_audit.get("error_counts"),
            "unique_strikes": xml_audit.get("unique_strikes"),
            "pitch_range": xml_audit.get("pitch_range"),
            "min_bpm": min(audit_qpm) if audit_qpm else None,
            "max_bpm": max(audit_qpm) if audit_qpm else None,
            "minimum_same_pitch_gap": xml_audit.get("minimum_same_pitch_gap"),
        },
        "midi_constraints": {
            "status": midi_constraints.get("status"),
            "ok": midi_constraints.get("ok"),
            "real_note_on_count": midi_constraints.get("real_note_on_count"),
            "unique_strike_count": midi_constraints.get("unique_strike_count"),
            "pitch_range": midi_constraints.get("pitch_range"),
            "minimum_same_pitch_gap_q": midi_constraints.get("minimum_same_pitch_gap_q"),
            "tempos": midi_constraints.get("tempos"),
            "errors": midi_constraints.get("errors"),
        },
    }

    # --- Layer 2: Multi-Representation Round-Trip & Fidelity Checks ---
    notation_diag = inspect_notation_durations(xml_path, input_id=xml_path.name)
    notation_ok = (
        notation_diag.get("status") in ("pass", "ok")
        and notation_diag.get("mismatch_count", -1) == 0
        and notation_diag.get("unsupported_count", -1) == 0
    )
    if not notation_ok:
        failure_reasons.append(
            f"layer2_notation_durations_failed:mismatches={notation_diag.get('mismatch_count')},unsupported={notation_diag.get('unsupported_count')}"
        )

    xml_transpose = summarize_xml_transposition(xml_path, input_id=xml_path.name)
    no_transpose_ok = xml_transpose.get("status") == "ok" and len(xml_transpose.get("transpositions", [])) == 0
    if not no_transpose_ok:
        failure_reasons.append("layer2_unexpected_transpose_tag_in_output_xml")

    ir_multiset = Counter((ev.midi, ev.onset_q, ev.duration_q) for ev in score.events)
    xml_ir_parity: dict[str, Any]
    try:
        xml_reloaded = load_score(xml_path)
        xml_multiset = Counter((ev.midi, ev.onset_q, ev.duration_q) for ev in xml_reloaded.events)
        xml_events_match = xml_multiset == ir_multiset
        xml_total_q_match = xml_reloaded.total_q == score.total_q
        xml_measures_match = [
            (m.measure, m.start_q, m.end_q, m.beats, m.beat_type) for m in xml_reloaded.measures
        ] == [
            (m.measure, m.start_q, m.end_q, m.beats, m.beat_type) for m in score.measures
        ]
        xml_tempos_match = [(t.onset_q, t.quarter_bpm) for t in xml_reloaded.tempos] == [
            (t.onset_q, t.quarter_bpm) for t in score.tempos
        ]
        xml_ir_ok = xml_events_match and xml_total_q_match and xml_measures_match and xml_tempos_match
        if not xml_ir_ok:
            diff_missing = list((ir_multiset - xml_multiset).elements())[:5]
            diff_extra = list((xml_multiset - ir_multiset).elements())[:5]
            failure_reasons.append(
                f"layer2_xml_ir_mismatch:events={xml_events_match},total_q={xml_total_q_match},measures={xml_measures_match},tempos={xml_tempos_match}"
            )
        else:
            diff_missing, diff_extra = [], []
        xml_ir_parity = {
            "status": "pass" if xml_ir_ok else "fail",
            "events_match": xml_events_match,
            "total_q_match": xml_total_q_match,
            "measures_match": xml_measures_match,
            "tempos_match": xml_tempos_match,
            "reloaded_event_count": len(xml_reloaded.events),
            "ir_event_count": len(score.events),
            "missing_sample": diff_missing,
            "extra_sample": diff_extra,
        }
    except Exception as exc:
        xml_ir_ok = False
        failure_reasons.append(f"layer2_xml_reload_exception:{type(exc).__name__}:{exc}")
        xml_ir_parity = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}

    midi_intervals_res = extract_midi_note_intervals(midi_path)
    if midi_intervals_res.get("status") != "ok":
        midi_ir_ok = False
        failure_reasons.append(f"layer2_midi_interval_parse_failed:{midi_intervals_res.get('reason')}")
        midi_ir_parity: dict[str, Any] = {"status": "fail", **midi_intervals_res}
    else:
        midi_multiset = Counter(
            (ev["midi"], ev["onset_q"], ev["duration_q"]) for ev in midi_intervals_res["intervals"]
        )
        midi_events_match = midi_multiset == ir_multiset
        midi_eot_match = midi_intervals_res["track0_eot_q"] == score.total_q
        midi_clean_off = (
            midi_intervals_res["unclosed_notes_count"] == 0
            and midi_intervals_res["orphan_note_off_count"] == 0
        )
        expected_tempos = [(t.onset_q, encode_tempo_us(t.quarter_bpm)[0]) for t in score.tempos]
        actual_tempos = [
            (te["abs_q"], te["us_per_quarter"]) for te in midi_intervals_res["tempo_events"]
        ]
        midi_tempos_match = actual_tempos == expected_tempos
        midi_ir_ok = midi_events_match and midi_eot_match and midi_clean_off and midi_tempos_match
        if not midi_ir_ok:
            failure_reasons.append(
                f"layer2_midi_ir_mismatch:events={midi_events_match},eot={midi_eot_match},clean_off={midi_clean_off},tempos={midi_tempos_match}"
            )
        midi_ir_parity = {
            "status": "pass" if midi_ir_ok else "fail",
            "events_match": midi_events_match,
            "track0_eot_matches_total_q": midi_eot_match,
            "clean_note_offs": midi_clean_off,
            "tempos_match": midi_tempos_match,
            "midi_interval_count": len(midi_intervals_res["intervals"]),
            "ir_event_count": len(score.events),
            "ppq": midi_intervals_res["ppq"],
            "expected_tempos": expected_tempos,
            "actual_tempos": actual_tempos,
        }

    xml_midi_cmp = compare_xml_and_midi(xml_path, midi_path, xml_id=xml_path.name, midi_id=midi_path.name)
    xml_midi_ok = xml_midi_cmp.get("status") == "pass"
    if not xml_midi_ok:
        failure_reasons.append(
            f"layer2_xml_midi_mismatch:extra={xml_midi_cmp.get('extra_in_midi_count')},missing={xml_midi_cmp.get('missing_in_midi_count')}"
        )

    try:
        m21_check = _music21_cross_check(xml_path, midi_path, score)
        m21_ok = m21_check["status"] == "pass"
        if not m21_ok:
            failure_reasons.append("layer2_music21_cross_check_mismatch")
    except Exception as exc:
        m21_ok = False
        m21_check = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        failure_reasons.append(f"layer2_music21_exception:{type(exc).__name__}:{exc}")

    layer2_ok = notation_ok and no_transpose_ok and xml_ir_ok and midi_ir_ok and xml_midi_ok and m21_ok
    layer2 = {
        "status": "pass" if layer2_ok else "fail",
        "notation_durations": {
            "status": notation_diag.get("status"),
            "mismatch_count": notation_diag.get("mismatch_count"),
            "unsupported_count": notation_diag.get("unsupported_count"),
        },
        "zero_transpose_tag": {
            "status": "pass" if no_transpose_ok else "fail",
            "transpositions": xml_transpose.get("transpositions"),
            "written_pitch_range": xml_transpose.get("written_pitch_range"),
            "sounding_pitch_range": xml_transpose.get("sounding_pitch_range"),
        },
        "xml_vs_ir": xml_ir_parity,
        "midi_vs_ir": midi_ir_parity,
        "xml_vs_midi": {
            "status": xml_midi_cmp.get("status"),
            "matched_count": xml_midi_cmp.get("matched_count"),
            "extra_in_midi_count": xml_midi_cmp.get("extra_in_midi_count"),
            "missing_in_midi_count": xml_midi_cmp.get("missing_in_midi_count"),
            "xml_duplicate_strikes_count": xml_midi_cmp.get("xml_duplicate_strikes_count"),
        },
        "music21_cross_check": m21_check,
    }

    # --- Layer 3: Objective Musical Summary (Listening Honest Not Evaluated) ---
    mel_evs = [e for e in score.events if e.role == "melody"]
    acc_evs = [e for e in score.events if e.role == "accompaniment"]
    mel_midis = [e.midi for e in mel_evs]
    acc_midis = [e.midi for e in acc_evs]
    m_count = max(1, len(score.measures))
    layer3 = {
        "listening_evaluation": "not_evaluated",
        "note": "Automated validation covers Layer 1 physical constraints and Layer 2 score/MIDI fidelity; subjective listening evaluation requires human playback.",
        "objective_stats": {
            "total_q": str(score.total_q),
            "measure_count": len(score.measures),
            "duration_seconds": _compute_duration_seconds(score),
            "total_events": len(score.events),
            "melody_count": len(mel_evs),
            "accompaniment_count": len(acc_evs),
            "accompaniment_notes_per_measure": round(len(acc_evs) / m_count, 2),
            "melody_pitch_range": [pitch_name(min(mel_midis)), pitch_name(max(mel_midis))] if mel_midis else None,
            "accompaniment_pitch_range": [pitch_name(min(acc_midis)), pitch_name(max(acc_midis))] if acc_midis else None,
            "melody_octave_adjusted_count": sum(1 for e in mel_evs if e.octave_adjusted),
            "duration_clamped_count": sum(1 for e in score.events if e.duration_clamped_from_q is not None),
            "export_serialization": score.metadata.get("export_serialization"),
        },
    }

    passed = layer1_ok and layer2_ok
    return serialize_rationals(
        {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "failure_reasons": failure_reasons,
            "files": {
                "musicxml": str(xml_path),
                "midi": str(midi_path),
            },
            "layer1_constraints": layer1,
            "layer2_fidelity": layer2,
            "layer3_musical_summary": layer3,
        }
    )
