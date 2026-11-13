"""MusicXML (via music21 10.5.2) and Format-1 Standard MIDI File exporter."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction as F
import math
from pathlib import Path
import struct
import sys
from typing import Any
import xml.etree.ElementTree as ET

from music21 import chord as m21_chord
from music21 import clef as m21_clef
from music21 import duration as m21_duration
from music21 import key as m21_key
from music21 import meter as m21_meter
from music21 import note as m21_note
from music21 import pitch as m21_pitch
from music21 import stream as m21_stream
from music21 import tempo as m21_tempo
from music21 import tie as m21_tie

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from inspect_notation_midi import _paths_refer_to_same_file  # noqa: E402

from .model import ALLOWED_MIDI, NoteEvent, ScoreIR, UnsupportedScoreError, serialize_rationals

ALLOWED_SET = frozenset(ALLOWED_MIDI)
MIN_BPM = F(100)
MAX_BPM = F(220)
MAX_PPQ = 32767


def _encode_vlq(value: int) -> bytes:
    """Encode non-negative integer as MIDI Variable Length Quantity."""
    if value < 0:
        raise ValueError(f"VLQ requires non-negative int, got {value}")
    buf = [value & 0x7F]
    value >>= 7
    while value > 0:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.reverse()
    return bytes(buf)


def encode_tempo_us(quarter_bpm: F) -> tuple[int, dict[str, Any]]:
    """Convert quarter BPM into 24-bit microseconds/quarter, preventing >220 BPM rounding overflow."""
    if quarter_bpm <= 0:
        raise ValueError(f"Tempo must be positive, got {quarter_bpm}")
    exact_us = F(60_000_000) / quarter_bpm
    cand_ceil = math.ceil(exact_us)
    written_us = cand_ceil
    serialized_bpm = F(60_000_000, written_us)
    # Guard bottom boundary (100 BPM): if ceil dips below 100 while floor fits in [100, 220], pick floor.
    if MIN_BPM <= quarter_bpm <= MAX_BPM and serialized_bpm < MIN_BPM:
        cand_floor = math.floor(exact_us)
        if cand_floor > 0:
            floor_bpm = F(60_000_000, cand_floor)
            if MIN_BPM <= floor_bpm <= MAX_BPM:
                written_us = cand_floor
                serialized_bpm = floor_bpm

    return written_us, {
        "target_bpm": str(quarter_bpm),
        "exact_us": float(exact_us),
        "written_us": written_us,
        "us_delta": round(float(written_us - exact_us), 6),
        "serialized_bpm": float(serialized_bpm),
    }


def _compute_exact_ppq(score: ScoreIR) -> int:
    """Compute LCM-aligned PPQ >= 480 representing every onset, duration end, tempo, and measure cleanly."""
    denoms: list[int] = [4]
    for ev in score.events:
        denoms.append(ev.onset_q.denominator)
        denoms.append((ev.onset_q + ev.duration_q).denominator)
    for t in score.tempos:
        denoms.append(t.onset_q.denominator)
    for m in score.measures:
        denoms.append(m.start_q.denominator)
        denoms.append(m.end_q.denominator)
    denoms.append(score.total_q.denominator)

    grid_lcm = math.lcm(*denoms)
    if grid_lcm > MAX_PPQ:
        raise UnsupportedScoreError(
            f"Timing denominator LCM {grid_lcm} exceeds MIDI 15-bit PPQ limit {MAX_PPQ}.",
            issues=[{"code": "ppq_lcm_overflow", "lcm": grid_lcm, "max_ppq": MAX_PPQ}],
        )
    mul = max(1, math.ceil(480 / grid_lcm))
    ppq = grid_lcm * mul
    while ppq > MAX_PPQ and mul > 1:
        mul -= 1
        ppq = grid_lcm * mul
    if ppq > MAX_PPQ:
        ppq = grid_lcm
    return ppq


def _build_track_chunk(events_with_abs_tick: list[tuple[int, int, bytes]], eot_tick: int) -> bytes:
    """Serialize sorted (tick, sort_prio, payload) MIDI messages into an MTrk chunk."""
    items = sorted(events_with_abs_tick, key=lambda x: (x[0], x[1]))
    body = bytearray()
    prev_tick = 0
    for tick, _prio, msg in items:
        delta = tick - prev_tick
        if delta < 0:
            raise AssertionError(f"Negative delta tick: {delta}")
        body.extend(_encode_vlq(delta))
        body.extend(msg)
        prev_tick = tick
    eot_delta = max(0, eot_tick - prev_tick)
    body.extend(_encode_vlq(eot_delta))
    body.extend(b"\xFF\x2F\x00")
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def _export_midi(score: ScoreIR, midi_path: Path) -> dict[str, Any]:
    """Write Format-1 Standard MIDI File directly with exact rational tick grid."""
    ppq = _compute_exact_ppq(score)

    def to_tick(q: F) -> int:
        val = q * ppq
        if val.denominator != 1:
            raise AssertionError(f"Non-integer tick for {q} at ppq={ppq}: {val}")
        return int(val)

    total_ticks = to_tick(score.total_q)

    # Track 0: track name, time signature events, exact tempo events.
    t0_msgs: list[tuple[int, int, bytes]] = []
    title_bytes = b"MusicBox Meta"
    t0_msgs.append((0, 0, b"\xFF\x03" + _encode_vlq(len(title_bytes)) + title_bytes))

    prev_sig: tuple[int, int] | None = None
    for m in score.measures:
        sig = (m.beats, m.beat_type)
        if sig != prev_sig:
            prev_sig = sig
            if m.beat_type > 0 and (m.beat_type & (m.beat_type - 1)) == 0:
                dd = int(math.log2(m.beat_type))
                t0_msgs.append((to_tick(m.start_q), 1, bytes([0xFF, 0x58, 0x04, m.beats & 0xFF, dd & 0xFF, 24, 8])))

    tempo_records: list[dict[str, Any]] = []
    for t in sorted(score.tempos, key=lambda x: x.onset_q):
        us_per_q, t_info = encode_tempo_us(t.quarter_bpm)
        t_tick = to_tick(t.onset_q)
        t_info["onset_q"] = str(t.onset_q)
        t_info["tick"] = t_tick
        tempo_records.append(t_info)
        t0_msgs.append((t_tick, 2, b"\xFF\x51\x03" + us_per_q.to_bytes(3, "big")))

    track_chunks = [_build_track_chunk(t0_msgs, total_ticks)]

    # Tracks 1..N: one track per part (P1 Melody on ch=0 vel=94, P2 Accompaniment on ch=1 vel=66/62).
    is_flowing = score.metadata.get("variant") == "flowing"
    for part_idx, part_id in enumerate(score.parts):
        ch = part_idx % 16
        if ch == 9:
            ch = 10  # Avoid GM percussion channel 10 (0-based 9).
        p_name = score.part_names.get(part_id, part_id).encode("utf-8")
        msgs: list[tuple[int, int, bytes]] = [
            (0, 0, b"\xFF\x03" + _encode_vlq(len(p_name)) + p_name),
            (0, 1, bytes([0xC0 | ch, 10])),  # GM Program 10: Music Box
        ]
        default_vel = 94 if part_id == "P1" else (62 if is_flowing else 68)
        for ev in score.events:
            if ev.part != part_id:
                continue
            on_tick = to_tick(ev.onset_q)
            off_tick = to_tick(ev.onset_q + ev.duration_q)
            if off_tick <= on_tick:
                raise AssertionError(f"Zero/negative MIDI duration tick on {ev.id}")
            vel = 94 if ev.role == "melody" else default_vel
            # Order note-off (prio 2) strictly before note-on (prio 3) at matching ticks.
            msgs.append((off_tick, 2, bytes([0x80 | ch, ev.midi & 0x7F, 0])))
            msgs.append((on_tick, 3, bytes([0x90 | ch, ev.midi & 0x7F, vel & 0x7F])))
        track_chunks.append(_build_track_chunk(msgs, total_ticks))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(track_chunks), ppq)
    midi_bytes = header + b"".join(track_chunks)
    midi_path.write_bytes(midi_bytes)
    return {
        "ppq": ppq,
        "format": 1,
        "num_tracks": len(track_chunks),
        "total_ticks": total_ticks,
        "tempos": tempo_records,
    }


def _split_duration_elements(
    proto_factory: Any,
    dur_q: F,
    tie_in: bool = False,
    tie_out: bool = False,
) -> list[Any]:
    """Build Music21 linked notation elements for exact dur_q, recursively splitting unlinked chunks."""
    if dur_q <= 0:
        return []

    def try_split(target_q: F) -> list[Any] | None:
        proto = proto_factory()
        proto.duration = m21_duration.Duration(target_q)
        pieces = proto.splitAtDurations()
        if pieces and all(getattr(p.duration, "linked", False) and F(p.duration.quarterLength) > 0 for p in pieces):
            total_got = sum(F(p.duration.quarterLength) for p in pieces)
            if total_got == target_q:
                return list(pieces)
        return None

    pieces = try_split(dur_q)
    if pieces is None:
        # Decompose into largest power-of-two fraction <= dur_q + remainder cleanly.
        power = F(4)
        while power > dur_q:
            power /= 2
        if power <= 0 or dur_q - power <= 0:
            raise UnsupportedScoreError(
                f"Cannot express exact duration {dur_q} as standard MusicXML linked notation.",
                issues=[{"code": "unrepresentable_notation_duration", "duration_q": str(dur_q)}],
            )
        pieces = try_split(power)
        rem_pieces = _split_duration_elements(proto_factory, dur_q - power, tie_in=False, tie_out=False)
        if pieces is None:
            raise UnsupportedScoreError(
                f"Failed to link power-of-two duration {power}.",
                issues=[{"code": "unrepresentable_notation_duration", "duration_q": str(dur_q)}],
            )
        pieces = pieces + rem_pieces

    n = len(pieces)
    for idx, piece in enumerate(pieces):
        if isinstance(piece, m21_note.Rest):
            continue
        has_prev = tie_in or (idx > 0)
        has_next = tie_out or (idx < n - 1)
        if has_prev and has_next:
            piece.tie = m21_tie.Tie("continue")
        elif has_next:
            piece.tie = m21_tie.Tie("start")
        elif has_prev:
            piece.tie = m21_tie.Tie("stop")
        else:
            piece.tie = None
    return pieces


def _make_pitch(midi_val: int) -> m21_pitch.Pitch:
    p = m21_pitch.Pitch(midi=midi_val)
    return p


def _split_rest_elements(rest_q: F, is_full_nominal_measure: bool) -> list[m21_note.Rest]:
    """Build standard linked Rest elements without emitting ambiguous partial-bar whole rests.

    music21 promotes partial-measure <rest><type>whole</type> (4q inside e.g. 4/2 or 3/2) to a
    full-measure breve/dotted-whole rest on re-parse, shifting all downstream measure offsets.
    Only emit an unsplit single rest when it represents an entire canonical measure; otherwise
    decompose spans >= 4q into <= 2q (half-rest) chunks plus standard linked remainder.
    """
    if rest_q <= 0:
        return []
    if is_full_nominal_measure:
        single = _split_duration_elements(lambda: m21_note.Rest(), rest_q)
        if len(single) == 1:
            return single
    pieces: list[m21_note.Rest] = []
    rem = rest_q
    while rem >= F(4):
        pieces.extend(_split_duration_elements(lambda: m21_note.Rest(), F(2)))
        rem -= F(2)
    if rem > 0:
        pieces.extend(_split_duration_elements(lambda: m21_note.Rest(), rem))
    return pieces


def _pack_part_voice_lanes(events: list[NoteEvent]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group simultaneous identical-(onset, dur) events into Chords and pack into non-overlapping lanes."""
    by_span: dict[tuple[F, F], list[int]] = defaultdict(list)
    for ev in events:
        by_span[(ev.onset_q, ev.duration_q)].append(ev.midi)

    items = [
        {"onset_q": on_q, "duration_q": dur_q, "end_q": on_q + dur_q, "midis": sorted(set(midis))}
        for (on_q, dur_q), midis in sorted(by_span.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]

    lanes: list[tuple[F, list[dict[str, Any]]]] = []
    for item in items:
        placed = False
        for idx, (lane_end, lane_items) in enumerate(lanes):
            if lane_end <= item["onset_q"]:
                lane_items.append(item)
                lanes[idx] = (item["end_q"], lane_items)
                placed = True
                break
        if not placed:
            lanes.append((item["end_q"], [item]))

    return [(str(idx), lane_items) for idx, (_end, lane_items) in enumerate(lanes, start=1)]


def _export_musicxml(score: ScoreIR, xml_path: Path) -> None:
    """Export ScoreIR to concert-pitch MusicXML using music21 10.5.2 and clean implicit flags."""
    m21_score = m21_stream.Score()
    tempos_by_q = defaultdict(list)
    for t in score.tempos:
        tempos_by_q[t.onset_q].append(t)

    part_events_map: dict[str, list[NoteEvent]] = defaultdict(list)
    for ev in score.events:
        part_events_map[ev.part].append(ev)

    implicit_indices: set[int] = set()
    measure_num_labels: list[str] = []
    for m in score.measures:
        measure_num_labels.append(m.measure)
        if m.implicit or m.length_q < m.nominal_q:
            implicit_indices.add(m.measure_index)

    for p_idx, part_id in enumerate(score.parts):
        part_stream = m21_stream.Part(id=part_id)
        part_stream.partName = score.part_names.get(part_id, part_id)
        lanes = _pack_part_voice_lanes(part_events_map.get(part_id, []))
        if not lanes:
            lanes = [("1", [])]

        # Pre-index slices per measure index and voice id.
        slices_by_mi_voice: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for voice_id, lane_items in lanes:
            for item in lane_items:
                item_start = item["onset_q"]
                item_end = item["end_q"]
                for m in score.measures:
                    ov_start = max(m.start_q, item_start)
                    ov_end = min(m.end_q, item_end)
                    if ov_start < ov_end:
                        slices_by_mi_voice[(m.measure_index, voice_id)].append(
                            {
                                "local_offset": ov_start - m.start_q,
                                "dur_q": ov_end - ov_start,
                                "midis": item["midis"],
                                "tie_in": item_start < m.start_q,
                                "tie_out": item_end > m.end_q,
                            }
                        )

        prev_sig: tuple[int, int] | None = None
        for m_idx, m in enumerate(score.measures):
            # Music21 requires numeric measure numbers or clean post-fixup; set m_idx + 1 then fix attribute in XML.
            m_stream = m21_stream.Measure(number=m_idx + 1)
            if m_idx == 0:
                m_stream.insert(0, m21_clef.TrebleClef())
                m_stream.insert(0, m21_key.KeySignature(0))
            sig = (m.beats, m.beat_type)
            if sig != prev_sig:
                prev_sig = sig
                m_stream.insert(0, m21_meter.TimeSignature(f"{m.beats}/{m.beat_type}"))

            if p_idx == 0:
                for t_ev in tempos_by_q.values():
                    for t in t_ev:
                        if m.start_q <= t.onset_q < m.end_q:
                            bpm_val = int(t.quarter_bpm) if t.quarter_bpm.denominator == 1 else round(float(t.quarter_bpm), 6)
                            m_stream.insert(t.onset_q - m.start_q, m21_tempo.MetronomeMark(number=bpm_val))

            active_voices = [
                vid for vid, _ in lanes
                if vid == "1" or slices_by_mi_voice.get((m.measure_index, vid))
            ]
            use_voice_wrappers = len(active_voices) > 1

            for vid in active_voices:
                target_container: m21_stream.Stream = m21_stream.Voice(id=vid) if use_voice_wrappers else m_stream
                cursor_q = F(0)
                m_slices = sorted(slices_by_mi_voice.get((m.measure_index, vid), []), key=lambda s: s["local_offset"])
                for sl in m_slices:
                    gap = sl["local_offset"] - cursor_q
                    if gap > 0:
                        rest_pieces = _split_rest_elements(gap, is_full_nominal_measure=False)
                        for rp in rest_pieces:
                            target_container.insert(cursor_q, rp)
                            cursor_q += F(rp.duration.quarterLength)
                    midis = sl["midis"]
                    if len(midis) == 1:
                        proto_fn = lambda m_val=midis[0]: m21_note.Note(_make_pitch(m_val))
                    else:
                        proto_fn = lambda m_vals=midis: m21_chord.Chord([_make_pitch(mv) for mv in m_vals])
                    note_pieces = _split_duration_elements(
                        proto_fn,
                        sl["dur_q"],
                        tie_in=sl["tie_in"],
                        tie_out=sl["tie_out"],
                    )
                    for np_el in note_pieces:
                        target_container.insert(cursor_q, np_el)
                        cursor_q += F(np_el.duration.quarterLength)

                trailing_gap = m.length_q - cursor_q
                if trailing_gap > 0:
                    is_full_nom = (
                        cursor_q == 0
                        and not m.implicit
                        and m.length_q == m.nominal_q
                    )
                    rest_pieces = _split_rest_elements(trailing_gap, is_full_nominal_measure=is_full_nom)
                    for rp in rest_pieces:
                        target_container.insert(cursor_q, rp)
                        cursor_q += F(rp.duration.quarterLength)

                if use_voice_wrappers:
                    m_stream.insert(0, target_container)

            part_stream.append(m_stream)
        m21_score.insert(0, part_stream)

    # Write MusicXML and fix measure number labels + implicit="yes" cleanly in UTF-8 XML.
    written = m21_score.write("musicxml", fp=str(xml_path))
    xml_bytes = Path(written).read_bytes()
    root = ET.fromstring(xml_bytes)
    for part_el in root.findall("part"):
        for idx, meas_el in enumerate(part_el.findall("measure")):
            if idx < len(measure_num_labels):
                meas_el.set("number", measure_num_labels[idx])
            if idx in implicit_indices:
                meas_el.set("implicit", "yes")
                # music21 pads short implicit measures up to nominal TimeSignature with invisible print-object="no" rests.
                for child in list(meas_el):
                    if child.tag == "note" and child.attrib.get("print-object") == "no" and child.find("rest") is not None:
                        meas_el.remove(child)
            elif "implicit" in meas_el.attrib:
                del meas_el.attrib["implicit"]

    ET.indent(root, space="  ")
    doctype = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    )
    xml_text = doctype + ET.tostring(root, encoding="unicode")
    xml_path.write_text(xml_text, encoding="utf-8")


def export_variant(score: ScoreIR, directory: Path | str, name: str) -> dict[str, Path]:
    """Export arranged ScoreIR variant as MusicXML and Format-1 MIDI without touching inputs."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{name}.musicxml"
    midi_path = out_dir / f"{name}.mid"

    if score.source_path is not None:
        src = Path(score.source_path)
        for tgt in (xml_path, midi_path):
            if _paths_refer_to_same_file(tgt, src):
                raise ValueError(f"Refusing to overwrite input file '{src}' with export '{tgt}'.")

    for ev in score.events:
        if ev.midi not in ALLOWED_SET:
            raise UnsupportedScoreError(
                f"Cannot export non-30-note MIDI pitch {ev.midi}.",
                issues=[{"code": "illegal_export_pitch", "midi": ev.midi, "event_id": ev.id}],
            )
        if ev.duration_q <= 0:
            raise UnsupportedScoreError(
                f"Cannot export non-positive duration {ev.duration_q} on {ev.id}.",
                issues=[{"code": "invalid_export_duration", "duration_q": str(ev.duration_q), "event_id": ev.id}],
            )

    _export_musicxml(score, xml_path)
    midi_meta = _export_midi(score, midi_path)
    score.metadata["export_serialization"] = serialize_rationals(midi_meta)
    return {"musicxml": xml_path, "midi": midi_path}
