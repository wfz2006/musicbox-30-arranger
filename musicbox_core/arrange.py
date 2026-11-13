"""Music box score analysis, constraint-preserving arrangement, and variant generation."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction as F
import math
from typing import Any

from .model import (
    ALLOWED_MIDI,
    MeasureSpan,
    NoteEvent,
    ScoreIR,
    SourceRef,
    TempoEvent,
    UnsupportedScoreError,
    parse_fraction,
    pitch_name,
    serialize_rationals,
)

ALLOWED_LIST: tuple[int, ...] = tuple(sorted(ALLOWED_MIDI))
ALLOWED_SET: frozenset[int] = frozenset(ALLOWED_MIDI)
MIN_BPM = F(100)
MAX_BPM = F(220)
DEFAULT_SCALES: tuple[F, ...] = (F(1, 2), F(1), F(2))
DEFAULT_TRANSPOSES: tuple[int, ...] = tuple(range(-5, 7))
VALID_BEAT_TYPES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

PITCH_CLASS_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

CHORD_TEMPLATES: tuple[tuple[str, tuple[int, ...], int], ...] = (
    ("maj", (0, 4, 7), 0),
    ("min", (0, 3, 7), 0),
    ("dom7", (0, 4, 7, 10), 1),
    ("min7", (0, 3, 7, 10), 1),
    ("maj7", (0, 4, 7, 11), 2),
    ("m7b5", (0, 3, 6, 10), 2),
    ("dim", (0, 3, 6), 2),
    ("dim7", (0, 3, 6, 9), 3),
    ("sus4", (0, 5, 7), 2),
    ("sus2", (0, 2, 7), 2),
    ("aug", (0, 4, 8), 3),
)


def _legal_octaves_for_midi(midi_val: int) -> list[int]:
    """Return sorted legal 30-note box pitches matching midi_val pitch class."""
    pc = midi_val % 12
    return [p for p in ALLOWED_LIST if p % 12 == pc]


def _closest_in_register(pc: int, low: int, high: int, target: int) -> int | None:
    """Return legal pitch matching pc closest to target within [low, high], falling back to full box."""
    in_band = [p for p in ALLOWED_LIST if p % 12 == pc % 12 and low <= p <= high]
    pool = in_band if in_band else _legal_octaves_for_midi(pc)
    if not pool:
        return None
    return min(pool, key=lambda p: (abs(p - target), p))


def _locate_measure_for_q(onset_q: F, measures: list[MeasureSpan]) -> MeasureSpan | None:
    for m in measures:
        if m.start_q <= onset_q < m.end_q:
            return m
    return measures[-1] if measures else None


def _scale_time_signature(beats: int, beat_type: int, nominal_q: F, scale: F) -> tuple[int, int] | None:
    """Compute a valid power-of-two music notation meter matching nominal_q * scale."""
    scaled_nom = nominal_q * scale
    if scaled_nom <= 0:
        return None
    # Prefer keeping numerator `beats` fixed and scaling `beat_type`.
    candidate_bt = F(beat_type) / scale
    if candidate_bt.denominator == 1 and int(candidate_bt) in VALID_BEAT_TYPES:
        return (beats, int(candidate_bt))
    # Fallback: scale numerator against valid power-of-two beat types cleanly.
    for bt in (4, 8, 2, 16, 1, 32, 64):
        b_exact = scaled_nom * F(bt, 4)
        if b_exact.denominator == 1 and 1 <= int(b_exact) <= 64:
            return (int(b_exact), bt)
    return None


def _infer_chord_for_window(
    pc_weights: dict[int, F],
    bass_midi: int | None,
) -> dict[str, Any] | None:
    """Infer triad/seventh chord label using pitch-class template weighting without assuming bass==root."""
    if not pc_weights:
        return None
    total_w = sum(pc_weights.values())
    if total_w <= 0:
        return None
    bass_pc = (bass_midi % 12) if bass_midi is not None else None
    best: dict[str, Any] | None = None

    for root in range(12):
        for quality, intervals, complexity_pen in CHORD_TEMPLATES:
            chord_pcs = {(root + iv) % 12 for iv in intervals}
            in_chord_w = sum(w for pc, w in pc_weights.items() if pc in chord_pcs)
            if in_chord_w <= 0:
                continue
            out_chord_w = total_w - in_chord_w
            root_w = pc_weights.get(root, F(0))
            fifth_pc = (root + 7) % 12 if 7 in intervals else None
            third_pc = (root + 4) % 12 if 4 in intervals else ((root + 3) % 12 if 3 in intervals else None)
            root_present = 1 if root in pc_weights else 0
            third_present = 1 if (third_pc is not None and third_pc in pc_weights) else 0
            fifth_present = 1 if (fifth_pc is not None and fifth_pc in pc_weights) else 0
            distinct_hit = sum(1 for pc in chord_pcs if pc in pc_weights)

            score = (
                float(in_chord_w / total_w) * 100.0
                - float(out_chord_w / total_w) * 45.0
                + distinct_hit * 12.0
                + root_present * 14.0
                + third_present * 10.0
                + fifth_present * 5.0
                + (float(root_w / total_w) * 15.0)
                - complexity_pen * 3.5
                + (4.0 if bass_pc == root else 0.0)
            )
            if best is None or score > best["score"]:
                best = {
                    "root_pc": root,
                    "quality": quality,
                    "label": f"{PITCH_CLASS_NAMES[root]}{'' if quality == 'maj' else ('m' if quality == 'min' else quality)}",
                    "chord_pcs": sorted(chord_pcs),
                    "bass_pc": bass_pc,
                    "bass_pitch": pitch_name(bass_midi) if bass_midi is not None else None,
                    "bass_matches_root": bass_pc == root if bass_pc is not None else None,
                    "confidence": round(max(0.0, min(1.0, float(in_chord_w / total_w) * (0.5 + 0.5 * distinct_hit / len(chord_pcs)))), 3),
                    "score": round(score, 3),
                }
    return best


def _decompose_score(
    score: ScoreIR,
    melody_part: str | None = None,
) -> tuple[dict[str, Any], list[NoteEvent], list[NoteEvent], list[dict[str, Any]], str]:
    """Separate score into primary melody strikes, accompaniment source pool, and harmony windows."""
    if not score.parts:
        raise UnsupportedScoreError("Score has no parts.", issues=[{"code": "empty_parts"}])

    events_by_part: dict[str, list[NoteEvent]] = defaultdict(list)
    for ev in score.events:
        events_by_part[ev.part].append(ev)

    if melody_part is not None:
        if melody_part not in score.parts:
            raise UnsupportedScoreError(
                f"Requested melody part '{melody_part}' not found in score parts {score.parts}.",
                issues=[{"code": "unknown_melody_part", "part": melody_part, "available_parts": list(score.parts)}],
            )
        if not events_by_part.get(melody_part):
            raise UnsupportedScoreError(
                f"Requested melody part '{melody_part}' contains no pitched notes.",
                issues=[{"code": "empty_melody_part", "part": melody_part}],
            )

    part_stats: list[dict[str, Any]] = []
    for idx, part_id in enumerate(score.parts):
        p_events = sorted(events_by_part.get(part_id, []), key=lambda e: (e.onset_q, e.midi))
        midis = sorted(e.midi for e in p_events)
        distinct_onsets = sorted({e.onset_q for e in p_events})
        median_midi = midis[len(midis) // 2] if midis else None
        mean_midi = (sum(midis) / len(midis)) if midis else None
        part_stats.append(
            {
                "part": part_id,
                "name": score.part_names.get(part_id, ""),
                "part_index": idx,
                "strike_event_count": len(p_events),
                "distinct_onset_count": len(distinct_onsets),
                "min_midi": midis[0] if midis else None,
                "median_midi": median_midi,
                "max_midi": midis[-1] if midis else None,
                "mean_midi": round(mean_midi, 2) if mean_midi is not None else None,
                "pitch_range": [pitch_name(midis[0]), pitch_name(midis[-1])] if midis else None,
            }
        )

    if melody_part is not None:
        chosen_part = melody_part
        selection_method = "user_specified"
    else:
        non_empty = [ps for ps in part_stats if ps["median_midi"] is not None]
        if not non_empty:
            raise UnsupportedScoreError("Score contains no pitched strike events.", issues=[{"code": "empty_score"}])
        winner = max(
            non_empty,
            key=lambda ps: (ps["median_midi"], ps["mean_midi"] or 0.0, -ps["part_index"]),
        )
        chosen_part = str(winner["part"])
        selection_method = "highest_median_pitch_part"

    # Extract highest pitch per distinct onset in the melody part; preserve lower simultaneous voices as accompaniment pool.
    melody_by_onset: dict[F, list[NoteEvent]] = defaultdict(list)
    accomp_pool: list[NoteEvent] = []
    for ev in sorted(score.events, key=lambda e: (e.onset_q, e.midi)):
        if ev.part == chosen_part:
            melody_by_onset[ev.onset_q].append(ev)
        else:
            accomp_pool.append(ev)

    melody_strikes: list[NoteEvent] = []
    for idx, onset_q in enumerate(sorted(melody_by_onset.keys()), start=1):
        at_onset = sorted(melody_by_onset[onset_q], key=lambda e: (-e.midi, e.voice))
        top_midi = at_onset[0].midi
        top_events = [e for e in at_onset if e.midi == top_midi]
        lower_events = [e for e in at_onset if e.midi < top_midi]
        merged_refs: list[SourceRef] = []
        for e in top_events:
            merged_refs.extend(e.source_refs)
        max_dur = max(e.duration_q for e in top_events)
        melody_strikes.append(
            NoteEvent(
                id=f"mel_src_{idx:05d}",
                midi=top_midi,
                onset_q=onset_q,
                duration_q=max_dur,
                role="melody",
                source_refs=merged_refs,
                part=chosen_part,
                voice=top_events[0].voice,
                staff=top_events[0].staff,
                original_midi=top_midi,
            )
        )
        accomp_pool.extend(lower_events)

    accomp_pool.sort(key=lambda e: (e.onset_q, e.midi))

    # Measure source melody same-pitch gaps.
    last_by_midi: dict[int, F] = {}
    mel_gaps: list[F] = []
    mel_self_conflicts = 0
    for ev in melody_strikes:
        if ev.midi in last_by_midi:
            gap = ev.onset_q - last_by_midi[ev.midi]
            mel_gaps.append(gap)
            if gap < F(1):
                mel_self_conflicts += 1
        last_by_midi[ev.midi] = ev.onset_q

    # Build half-measure harmony windows using non-melody accompaniment pool (supplemented by melody if needed).
    harmony_windows: list[dict[str, Any]] = []
    use_combined_pc = len(score.parts) == 1 and not accomp_pool
    for m in score.measures:
        if m.length_q >= F(2):
            mid_q = m.start_q + m.length_q / 2
            sub_spans = [(m.start_q, mid_q, "H1"), (mid_q, m.end_q, "H2")]
        else:
            sub_spans = [(m.start_q, m.end_q, "H1")]

        for w_start, w_end, half_tag in sub_spans:
            acc_weights: dict[int, F] = defaultdict(lambda: F(0))
            all_weights: dict[int, F] = defaultdict(lambda: F(0))
            bass_candidates: list[int] = []
            for ev in accomp_pool:
                ev_end = ev.onset_q + ev.duration_q
                overlap = min(w_end, ev_end) - max(w_start, ev.onset_q)
                if overlap > 0:
                    acc_weights[ev.midi % 12] += overlap
                    all_weights[ev.midi % 12] += overlap
                    bass_candidates.append(ev.midi)
            for ev in melody_strikes:
                ev_end = ev.onset_q + ev.duration_q
                overlap = min(w_end, ev_end) - max(w_start, ev.onset_q)
                if overlap > 0:
                    all_weights[ev.midi % 12] += overlap * F(1, 2)

            if acc_weights and not use_combined_pc:
                weights = acc_weights
                basis = "accompaniment_events"
            else:
                weights = all_weights
                basis = "accompaniment_plus_melody_fallback"

            bass_midi = min(bass_candidates) if bass_candidates else None
            chord = _infer_chord_for_window(weights, bass_midi=bass_midi)
            harmony_windows.append(
                {
                    "measure_index": m.measure_index,
                    "measure": m.measure,
                    "half": half_tag,
                    "start_q": w_start,
                    "end_q": w_end,
                    "basis": basis,
                    "bass_midi": bass_midi,
                    "chord": chord,
                }
            )

    tempos_qpm = [t.quarter_bpm for t in score.tempos]
    time_sigs = sorted({f"{m.beats}/{m.beat_type}" for m in score.measures})
    labeled_windows = [w for w in harmony_windows if w["chord"] is not None]
    bass_not_root = sum(
        1 for w in labeled_windows if w["chord"] and w["chord"].get("bass_matches_root") is False
    )

    mel_midis = [e.midi for e in melody_strikes]
    analysis = {
        "source_path": score.source_path,
        "source_sha256": score.source_sha256,
        "total_q": score.total_q,
        "measure_count": len(score.measures),
        "part_stats": part_stats,
        "selected_melody_part": chosen_part,
        "melody_selection_method": selection_method,
        "melody": {
            "strike_count": len(melody_strikes),
            "pitch_range": [pitch_name(min(mel_midis)), pitch_name(max(mel_midis))] if mel_midis else None,
            "min_same_pitch_gap_q": min(mel_gaps) if mel_gaps else None,
            "source_self_conflict_count": mel_self_conflicts,
            "extraction_rule": "highest_pitch_per_distinct_onset_preserving_all_top_source_refs",
        },
        "accompaniment_pool_count": len(accomp_pool),
        "tempos": {
            "count": len(score.tempos),
            "min_bpm": min(tempos_qpm) if tempos_qpm else None,
            "max_bpm": max(tempos_qpm) if tempos_qpm else None,
            "events": [t.to_dict() for t in score.tempos],
        },
        "time_signatures": time_sigs,
        "harmony_summary": {
            "window_count": len(harmony_windows),
            "labeled_window_count": len(labeled_windows),
            "inversions_or_non_root_bass_windows": bass_not_root,
            "windows": [
                {
                    "measure_index": w["measure_index"],
                    "measure": w["measure"],
                    "half": w["half"],
                    "start_q": w["start_q"],
                    "end_q": w["end_q"],
                    "basis": w["basis"],
                    "bass_midi": w["bass_midi"],
                    "chord_label": w["chord"]["label"] if w["chord"] else None,
                    "chord_pcs": w["chord"]["chord_pcs"] if w["chord"] else [],
                    "confidence": w["chord"]["confidence"] if w["chord"] else 0.0,
                }
                for w in harmony_windows
            ],
        },
        "warnings": list(score.warnings),
        "heuristics_and_limitations": [
            "Melody auto-selection chooses the part with highest median strike MIDI (override with --melody-part).",
            "At polyphonic melody onsets, only the highest pitch is treated as primary melody; simultaneous lower notes join the accompaniment pool.",
            "Harmony inference scores pitch-class triad/seventh templates over half-measure windows and explicitly avoids assuming lowest bass equals chord root.",
        ],
    }
    return serialize_rationals(analysis), melody_strikes, accomp_pool, harmony_windows, chosen_part


def analyze_score(score: ScoreIR, melody_part: str | None = None) -> dict[str, Any]:
    """Analyze ScoreIR into part stats, melody strikes, tempo profile, and harmony windows."""
    analysis, _, _, _, _ = _decompose_score(score, melody_part=melody_part)
    return analysis


def _beam_search_melody(
    melody_strikes: list[NoteEvent],
    scale: F,
    transpose: int,
    measures: list[MeasureSpan],
    beam_width: int = 64,
) -> dict[str, Any]:
    """Search octave assignments keeping all melody strikes in 30-note box with >=1q same-pitch gap."""
    # Beam state dict keyed by (last_choice, active_cooldowns_tuple) -> (cost, chosen_indices, oct_adj_count)
    beam: dict[tuple[int | None, tuple[tuple[int, F], ...]], tuple[int, tuple[int, ...], int]] = {
        (None, ()): (0, (), 0)
    }

    for step_idx, ev in enumerate(melody_strikes):
        t_q = ev.onset_q * scale
        raw_p = ev.midi + transpose
        choices = _legal_octaves_for_midi(raw_p)
        prev_raw = (melody_strikes[step_idx - 1].midi + transpose) if step_idx > 0 else raw_p
        next_beam: dict[tuple[int | None, tuple[tuple[int, F], ...]], tuple[int, tuple[int, ...], int]] = {}

        for (prev_choice, cooldowns), (cost_so_far, path, adj_so_far) in beam.items():
            # Retain only active pitches struck strictly inside (t_q - 1, t_q].
            cd_map = {p: t for p, t in cooldowns if (t_q - t) < F(1)}
            for ch in choices:
                if ch in cd_map:
                    continue
                oct_diff = abs(ch - raw_p) // 12
                step_cost = oct_diff * 120 + (40 if oct_diff > 1 else 0)
                if ch < 60:
                    step_cost += 60
                elif ch < 67:
                    step_cost += 15
                if prev_choice is not None:
                    actual_interval = ch - prev_choice
                    orig_interval = raw_p - prev_raw
                    step_cost += abs(actual_interval - orig_interval) * 15
                    if abs(actual_interval) > 12:
                        step_cost += (abs(actual_interval) - 12) * 10
                new_cost = cost_so_far + step_cost
                new_adj = adj_so_far + (1 if ch != raw_p else 0)
                new_cd = dict(cd_map)
                new_cd[ch] = t_q
                state_key = (ch, tuple(sorted(new_cd.items())))
                existing = next_beam.get(state_key)
                if existing is None or new_cost < existing[0]:
                    next_beam[state_key] = (new_cost, path + (ch,), new_adj)

        if not next_beam:
            ref = ev.source_refs[0] if ev.source_refs else None
            m_span = _locate_measure_for_q(ev.onset_q, measures)
            return {
                "feasible": False,
                "failed_step_index": step_idx,
                "conflict_location": {
                    "part": ref.part if ref else ev.part,
                    "measure": ref.measure if ref else (m_span.measure if m_span else None),
                    "measure_index": ref.measure_index if ref else (m_span.measure_index if m_span else None),
                    "source_onset_q": str(ev.onset_q),
                    "scaled_onset_q": str(t_q),
                    "raw_transposed_midi": raw_p,
                    "raw_transposed_pitch": pitch_name(raw_p),
                    "legal_octaves": [pitch_name(p) for p in choices],
                    "reason": (
                        "All legal octaves for this melody pitch conflict within <1q of a preceding melody strike."
                        if choices
                        else "No matching pitch class exists in allowed 30 pitches."
                    ),
                },
            }

        sorted_states = sorted(next_beam.items(), key=lambda kv: (kv[1][0], kv[1][2], kv[1][1]))[:beam_width]
        beam = dict(sorted_states)

    best_key, (best_cost, best_path, best_adj) = min(beam.items(), key=lambda kv: (kv[1][0], kv[1][2], kv[1][1]))
    return {
        "feasible": True,
        "cost": best_cost,
        "octave_adjusted_count": best_adj,
        "assigned_pitches": list(best_path),
    }


def _build_melody_events(
    melody_strikes: list[NoteEvent],
    assigned_pitches: list[int],
    scale: F,
    transpose: int,
    scaled_total_q: F,
) -> tuple[list[NoteEvent], list[dict[str, Any]]]:
    """Materialize scaled, legality-verified melody NoteEvents and change records."""
    events: list[NoteEvent] = []
    adj_records: list[dict[str, Any]] = []
    for idx, (src_ev, out_midi) in enumerate(zip(melody_strikes, assigned_pitches), start=1):
        onset_q = src_ev.onset_q * scale
        dur_q = min(src_ev.duration_q * scale, scaled_total_q - onset_q)
        if dur_q <= 0:
            dur_q = F(1, 4)
        transposed_raw = src_ev.midi + transpose
        octave_adj = out_midi != transposed_raw
        reason: str | None = None
        if octave_adj:
            if transposed_raw not in ALLOWED_SET:
                reason = f"pitch_{pitch_name(transposed_raw)}_outside_30_notes_shifted_octave"
            else:
                reason = f"shifted_octave_to_avoid_1q_conflict_or_register_collision"
            ref = src_ev.source_refs[0] if src_ev.source_refs else None
            adj_records.append(
                {
                    "event_id": f"mel_{idx:05d}",
                    "part": src_ev.part,
                    "measure": ref.measure if ref else None,
                    "source_onset_q": str(src_ev.onset_q),
                    "target_onset_q": str(onset_q),
                    "source_midi": src_ev.midi,
                    "source_pitch": pitch_name(src_ev.midi),
                    "transposed_raw_midi": transposed_raw,
                    "assigned_midi": out_midi,
                    "assigned_pitch": pitch_name(out_midi),
                    "reason": reason,
                }
            )
        events.append(
            NoteEvent(
                id=f"mel_{idx:05d}",
                midi=out_midi,
                onset_q=onset_q,
                duration_q=dur_q,
                role="melody",
                source_refs=list(src_ev.source_refs),
                part="P1",
                voice="1",
                staff="1",
                original_midi=src_ev.midi,
                octave_adjusted=octave_adj,
                adjustment_reason=reason,
            )
        )
    return events, adj_records


def _can_strike(midi_val: int, onset_q: F, strike_map: dict[int, list[F]]) -> bool:
    for t in strike_map.get(midi_val, ()):
        if abs(t - onset_q) < F(1):
            return False
    return True


def _record_strike(midi_val: int, onset_q: F, strike_map: dict[int, list[F]]) -> None:
    strike_map.setdefault(midi_val, []).append(onset_q)


def _pc_names(pc_nums: list[int] | set[int]) -> list[str]:
    return [PITCH_CLASS_NAMES[pc % 12] for pc in sorted({pc % 12 for pc in pc_nums})]


def _window_harmony_candidates(
    window: dict[str, Any],
    accomp_pool: list[NoteEvent],
    transpose: int,
    local_melody_min: int | None,
) -> dict[str, Any]:
    """Build legal 30-note bass/upper candidates strictly from source accompaniment pitch classes in window."""
    src_start = window["start_q"]
    src_end = window["end_q"]
    overlapping = [
        ev
        for ev in accomp_pool
        if ev.onset_q < src_end and (ev.onset_q + ev.duration_q) > src_start
    ]
    window_refs: list[SourceRef] = []
    for ev in overlapping:
        window_refs.extend(ev.source_refs)

    raw_source_pc_nums = sorted({ev.midi % 12 for ev in overlapping})
    transposed_source_pc_nums = sorted({(pc + transpose) % 12 for pc in raw_source_pc_nums})
    allowed_pc_set = frozenset(transposed_source_pc_nums)

    if not overlapping or not allowed_pc_set:
        return {
            "bass_candidates": [],
            "upper_candidates": [],
            "source_refs": [],
            "raw_source_pc_nums": [],
            "transposed_source_pc_nums": [],
        }

    upper_ceiling = min(80, (local_melody_min - 2) if local_melody_min is not None else 76)
    if upper_ceiling < 62:
        upper_ceiling = 69

    bass_candidates: list[int] = []
    upper_candidates: list[int] = []

    def add_bass(p: int | None) -> None:
        if p is not None and p in ALLOWED_SET and (p % 12) in allowed_pc_set and p not in bass_candidates:
            bass_candidates.append(p)

    def add_upper(p: int | None) -> None:
        if p is not None and p in ALLOWED_SET and (p % 12) in allowed_pc_set and p not in upper_candidates:
            upper_candidates.append(p)

    # 1. Preserve lowest overlapping source bass pitch class first (register-shifted cleanly if needed).
    bass_src = min(ev.midi for ev in overlapping) + transpose
    add_bass(_closest_in_register(bass_src % 12, 53, min(69, upper_ceiling), target=max(53, min(65, bass_src))))

    # 2. Optional chord-template root preference ONLY if root is actually present in transposed source PCs.
    chord = window.get("chord")
    if chord is not None:
        template_root_tr = (int(chord["root_pc"]) + transpose) % 12
        if template_root_tr in allowed_pc_set:
            add_bass(_closest_in_register(template_root_tr, 53, min(67, upper_ceiling), target=60))

    # 3. Any remaining source PCs can fall back across legal bass/mid octaves (never outside allowed_pc_set).
    for pc in transposed_source_pc_nums:
        add_bass(_closest_in_register(pc, 53, min(67, upper_ceiling), target=60))
    for p in ALLOWED_LIST:
        if p <= 72 and (p % 12) in allowed_pc_set:
            add_bass(p)

    # 4. Upper candidates preserve source voicing register first, then legal octave fallbacks of source PCs.
    for ev in sorted(overlapping, key=lambda e: e.midi):
        up_src = ev.midi + transpose
        add_upper(_closest_in_register(up_src % 12, 60, upper_ceiling, target=max(62, min(74, up_src))))

    target_mid = max(62, min(70, upper_ceiling))
    for pc in transposed_source_pc_nums:
        add_upper(_closest_in_register(pc, 60, upper_ceiling, target=target_mid))
    for p in ALLOWED_LIST:
        if (p % 12) in allowed_pc_set and p >= 60:
            add_upper(p)
    for p in ALLOWED_LIST:
        if (p % 12) in allowed_pc_set:
            add_upper(p)

    return {
        "bass_candidates": bass_candidates,
        "upper_candidates": upper_candidates,
        "source_refs": window_refs,
        "raw_source_pc_nums": raw_source_pc_nums,
        "transposed_source_pc_nums": transposed_source_pc_nums,
    }


def _generate_accompaniment(
    texture: str,
    target_measures: list[MeasureSpan],
    harmony_windows: list[dict[str, Any]],
    accomp_pool: list[NoteEvent],
    melody_events: list[NoteEvent],
    scale: F,
    transpose: int,
    scaled_total_q: F,
) -> tuple[list[NoteEvent], dict[str, Any]]:
    """Build sparse or flowing accompaniment on already-scaled target_measures using only source PCs."""
    strike_map: dict[int, list[F]] = defaultdict(list)
    for m_ev in melody_events:
        strike_map[m_ev.midi].append(m_ev.onset_q)

    # Precompute static per-window source metadata for exact decision tracing.
    win_traces: list[dict[str, Any]] = []
    for w in harmony_windows:
        info = _window_harmony_candidates(w, accomp_pool, transpose, local_melody_min=None)
        win_traces.append(
            {
                "measure_index": w["measure_index"],
                "measure": w["measure"],
                "half": w["half"],
                "source_start_q": w["start_q"],
                "source_end_q": w["end_q"],
                "target_start_q": w["start_q"] * scale,
                "target_end_q": w["end_q"] * scale,
                "inferred_chord_label": w["chord"]["label"] if w.get("chord") else None,
                "raw_source_pc_nums": info["raw_source_pc_nums"],
                "transposed_source_pc_nums": info["transposed_source_pc_nums"],
                "source_refs": [
                    {
                        "part": ref.part,
                        "measure": ref.measure,
                        "note_index": ref.note_index,
                        "pitch": ref.pitch,
                        "onset_q": str(ref.onset_q) if ref.onset_q is not None else None,
                    }
                    for ref in info["source_refs"][:12]
                ],
                "slots_evaluated": [],
            }
        )

    def find_window_index(target_q: F) -> int | None:
        src_q = target_q / scale
        for idx_w, w in enumerate(harmony_windows):
            if w["start_q"] <= src_q < w["end_q"]:
                return idx_w
        return (len(harmony_windows) - 1) if harmony_windows else None

    def local_melody_band(slot_start: F, slot_end: F) -> tuple[int | None, int]:
        nearby_count = 0
        min_pitch: int | None = None
        for m_ev in melody_events:
            if (slot_start - F(1)) <= m_ev.onset_q < (slot_start + F(1)):
                nearby_count += 1
            if (slot_start - F(1, 2)) <= m_ev.onset_q < (slot_end + F(1, 2)):
                if min_pitch is None or m_ev.midi < min_pitch:
                    min_pitch = m_ev.midi
        return min_pitch, nearby_count

    def explain_blocked(pool: list[int], slot_t: F, min_mel: int | None, allow_above_mel: bool) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for p in pool[:8]:
            reasons: list[str] = []
            if not allow_above_mel and min_mel is not None and p >= min_mel:
                reasons.append(f"at_or_above_melody_{pitch_name(min_mel)}")
            collisions = [str(t) for t in strike_map.get(p, ()) if abs(t - slot_t) < F(1)]
            if collisions:
                reasons.append(f"same_pitch_gap_lt_1q_near_{','.join(collisions[:3])}")
            details.append({"pitch": pitch_name(p), "midi": p, "reasons": reasons or ["not_selected"]})
        return details

    acc_events: list[NoteEvent] = []
    skipped_conflict_slots = 0
    skipped_density_slots = 0
    skipped_empty_source_slots = 0
    placed_idx = 0

    for m in target_measures:
        # Target measures are ALREADY in target quarter-note time; do not multiply by scale again.
        m_start = m.start_q
        m_end = m.end_q
        m_len = m_end - m_start
        if m_len <= 0:
            continue

        if texture == "sparse":
            slots: list[tuple[F, F, str]] = []
            if m_len >= F(4):
                half = m_len / 2
                slots.append((m_start, half, "bass_and_top"))
                slots.append((m_start + half, m_end - (m_start + half), "upper_only"))
            elif m_len >= F(2):
                half = (m_len * F(2, 3)) if m.beats == 3 else (m_len / 2)
                slots.append((m_start, half, "bass_only"))
                slots.append((m_start + half, m_end - (m_start + half), "upper_only"))
            else:
                slots.append((m_start, m_len, "bass_only"))

            for slot_t, slot_dur, mode in slots:
                slot_dur = min(slot_dur, scaled_total_q - slot_t)
                if slot_dur <= 0:
                    continue
                w_idx = find_window_index(slot_t)
                if w_idx is None:
                    continue
                win = harmony_windows[w_idx]
                w_trace = win_traces[w_idx]
                min_mel, _ = local_melody_band(slot_t, slot_t + slot_dur)
                cand = _window_harmony_candidates(win, accomp_pool, transpose, min_mel)
                bass_cands = cand["bass_candidates"]
                up_cands = cand["upper_candidates"]
                refs = cand["source_refs"]

                if not bass_cands and not up_cands:
                    skipped_empty_source_slots += 1
                    w_trace["slots_evaluated"].append(
                        {"slot_q": str(slot_t), "mode": mode, "status": "skipped_source_empty"}
                    )
                    continue

                placed_pitches: list[int] = []
                if mode in ("bass_only", "bass_and_top"):
                    for b in bass_cands:
                        if _can_strike(b, slot_t, strike_map):
                            placed_idx += 1
                            acc_events.append(
                                NoteEvent(
                                    id=f"acc_{placed_idx:05d}",
                                    midi=b,
                                    onset_q=slot_t,
                                    duration_q=slot_dur,
                                    role="accompaniment",
                                    source_refs=refs[:2],
                                    part="P2",
                                    voice="1",
                                    staff="1",
                                )
                            )
                            _record_strike(b, slot_t, strike_map)
                            placed_pitches.append(b)
                            break
                if mode in ("upper_only", "bass_and_top"):
                    for u in up_cands:
                        if min_mel is not None and u >= min_mel:
                            continue
                        if _can_strike(u, slot_t, strike_map):
                            placed_idx += 1
                            acc_events.append(
                                NoteEvent(
                                    id=f"acc_{placed_idx:05d}",
                                    midi=u,
                                    onset_q=slot_t,
                                    duration_q=slot_dur,
                                    role="accompaniment",
                                    source_refs=refs[:2],
                                    part="P2",
                                    voice="1",
                                    staff="1",
                                )
                            )
                            _record_strike(u, slot_t, strike_map)
                            placed_pitches.append(u)
                            break

                if placed_pitches:
                    w_trace["slots_evaluated"].append(
                        {
                            "slot_q": str(slot_t),
                            "mode": mode,
                            "status": "placed",
                            "pitches": [pitch_name(p) for p in placed_pitches],
                        }
                    )
                else:
                    skipped_conflict_slots += 1
                    pool_tried = bass_cands if mode == "bass_only" else (up_cands if mode == "upper_only" else bass_cands + up_cands)
                    w_trace["slots_evaluated"].append(
                        {
                            "slot_q": str(slot_t),
                            "mode": mode,
                            "status": "skipped_conflict",
                            "blocked": explain_blocked(pool_tried, slot_t, min_mel, allow_above_mel=(mode == "bass_only")),
                        }
                    )

        else:
            step_q = F(1) if m_len >= F(5) else F(1, 2)
            slot_times: list[F] = []
            cur = m_start
            while cur < m_end:
                slot_times.append(cur)
                nxt = cur + step_q
                if step_q == F(1):
                    mid_probe = cur + F(1, 2)
                    has_mel_around_mid = any(
                        (cur - F(1, 4)) <= me.onset_q < (nxt + F(1, 4)) for me in melody_events
                    )
                    if not has_mel_around_mid and mid_probe < m_end:
                        slot_times.append(mid_probe)
                cur = nxt

            for idx_in_bar, slot_t in enumerate(slot_times):
                next_t = slot_times[idx_in_bar + 1] if idx_in_bar + 1 < len(slot_times) else m_end
                slot_dur = min(max(F(1, 2), next_t - slot_t), scaled_total_q - slot_t)
                if slot_dur <= 0:
                    continue
                w_idx = find_window_index(slot_t)
                if w_idx is None:
                    continue
                win = harmony_windows[w_idx]
                w_trace = win_traces[w_idx]
                min_mel, nearby_mel = local_melody_band(slot_t, slot_t + step_q)
                is_strong = (slot_t == m_start) or (m_len >= F(2) and slot_t == m_start + m_len / 2)

                cand = _window_harmony_candidates(win, accomp_pool, transpose, min_mel)
                bass_cands = cand["bass_candidates"]
                up_cands = cand["upper_candidates"]
                refs = cand["source_refs"]

                if not bass_cands and not up_cands:
                    skipped_empty_source_slots += 1
                    w_trace["slots_evaluated"].append(
                        {"slot_q": str(slot_t), "status": "skipped_source_empty"}
                    )
                    continue

                if nearby_mel >= 2 and not is_strong:
                    skipped_density_slots += 1
                    w_trace["slots_evaluated"].append(
                        {
                            "slot_q": str(slot_t),
                            "status": "skipped_melody_density",
                            "nearby_melody_count": nearby_mel,
                        }
                    )
                    continue

                ordered_up = sorted(up_cands)
                high_up = list(reversed(ordered_up))
                mid_up = ordered_up[len(ordered_up) // 2 :] + ordered_up[: len(ordered_up) // 2]

                if slot_t == m_start:
                    cand_pool = bass_cands + ordered_up
                    hold_dur = min(max(slot_dur, m_len / 2), scaled_total_q - slot_t)
                elif idx_in_bar % 4 == 1:
                    cand_pool = ordered_up + bass_cands
                    hold_dur = min(max(slot_dur, step_q * 2), scaled_total_q - slot_t)
                elif idx_in_bar % 4 == 2:
                    cand_pool = high_up + ordered_up + bass_cands
                    hold_dur = min(max(slot_dur, step_q * 2), scaled_total_q - slot_t)
                else:
                    cand_pool = mid_up + bass_cands
                    hold_dur = slot_dur

                picked = None
                for p in cand_pool:
                    if slot_t != m_start and min_mel is not None and p >= min_mel and p not in bass_cands:
                        continue
                    if _can_strike(p, slot_t, strike_map):
                        picked = p
                        break
                if picked is None:
                    for p in bass_cands + ordered_up:
                        if _can_strike(p, slot_t, strike_map):
                            picked = p
                            break

                if picked is not None:
                    placed_idx += 1
                    acc_events.append(
                        NoteEvent(
                            id=f"acc_{placed_idx:05d}",
                            midi=picked,
                            onset_q=slot_t,
                            duration_q=hold_dur,
                            role="accompaniment",
                            source_refs=refs[:2],
                            part="P2",
                            voice="1",
                            staff="1",
                        )
                    )
                    _record_strike(picked, slot_t, strike_map)
                    w_trace["slots_evaluated"].append(
                        {"slot_q": str(slot_t), "status": "placed", "pitches": [pitch_name(picked)]}
                    )
                else:
                    skipped_conflict_slots += 1
                    w_trace["slots_evaluated"].append(
                        {
                            "slot_q": str(slot_t),
                            "status": "skipped_conflict",
                            "blocked": explain_blocked(cand_pool, slot_t, min_mel, allow_above_mel=True),
                        }
                    )

    stats = {
        "texture": texture,
        "generated_event_count": len(acc_events),
        "skipped_conflict_slots": skipped_conflict_slots,
        "skipped_melody_density_slots": skipped_density_slots,
        "skipped_empty_source_slots": skipped_empty_source_slots,
        "window_traces": win_traces,
    }
    return acc_events, stats


def _build_per_window_audit(
    final_events: list[NoteEvent],
    window_traces: list[dict[str, Any]],
    texture: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconcile finalized accompaniment events against each source harmony window without mixing sustained notes."""
    acc_events = [e for e in final_events if e.role == "accompaniment"]
    window_audits: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = defaultdict(int)
    windows_with_new = 0
    windows_without_new = 0
    windows_with_added = 0

    for wt in window_traces:
        t_start: F = wt["target_start_q"]
        t_end: F = wt["target_end_q"]
        src_tr_nums = set(wt["transposed_source_pc_nums"])

        new_strikes = [e for e in acc_events if t_start <= e.onset_q < t_end]
        sustained_in = [e for e in acc_events if e.onset_q < t_start and (e.onset_q + e.duration_q) > t_start]

        new_pc_nums = {e.midi % 12 for e in new_strikes}
        sustained_pc_nums = {e.midi % 12 for e in sustained_in}
        missing_pc_nums = src_tr_nums - new_pc_nums
        added_pc_nums = new_pc_nums - src_tr_nums

        slots = wt["slots_evaluated"]
        slot_statuses = [s["status"] for s in slots]

        if new_strikes:
            windows_with_new += 1
            no_new_reason = None
            if not missing_pc_nums:
                sel_reason = "all_transposed_source_pcs_struck"
            elif texture == "sparse":
                sel_reason = "sparse_voicing_selected_subset_from_source_pcs"
            elif "skipped_melody_density" in slot_statuses or "skipped_conflict" in slot_statuses:
                sel_reason = "flowing_subset_due_to_density_or_1q_conflict"
            else:
                sel_reason = "flowing_selected_subset_from_source_pcs"
        else:
            windows_without_new += 1
            reasons: list[str] = []
            if not src_tr_nums:
                reasons.append("source_empty_window")
            else:
                if "skipped_melody_density" in slot_statuses:
                    reasons.append("melody_density_thinning")
                if "skipped_conflict" in slot_statuses:
                    reasons.append("1q_conflict_or_register")
                if not slots:
                    reasons.append("texture_sparse_omit" if texture == "sparse" else "no_target_slot_in_window")
            no_new_reason = "+".join(reasons) if reasons else "unplaced"
            sel_reason = no_new_reason
            reason_counts[no_new_reason] += 1

        if added_pc_nums:
            windows_with_added += 1

        window_audits.append(
            {
                "measure_index": wt["measure_index"],
                "measure": wt["measure"],
                "half": wt["half"],
                "source_span_q": [str(wt["source_start_q"]), str(wt["source_end_q"])],
                "target_span_q": [str(t_start), str(t_end)],
                "inferred_chord_label": wt["inferred_chord_label"],
                "source_raw_pcs": _pc_names(wt["raw_source_pc_nums"]),
                "transposed_source_pcs": _pc_names(src_tr_nums),
                "new_strike_pcs": _pc_names(new_pc_nums),
                "sustained_into_window_pcs": _pc_names(sustained_pc_nums),
                "missing_pcs": _pc_names(missing_pc_nums),
                "added_pcs": _pc_names(added_pc_nums),
                "output_event_ids": [e.id for e in new_strikes],
                "new_strike_pitches": [
                    {"id": e.id, "midi": e.midi, "pitch": pitch_name(e.midi), "onset_q": str(e.onset_q), "duration_q": str(e.duration_q)}
                    for e in new_strikes
                ],
                "sustained_into_window_event_ids": [e.id for e in sustained_in],
                "source_refs": wt["source_refs"],
                "selection_reason": sel_reason,
                "no_new_strike_reason": no_new_reason,
                "slot_decisions": slots,
            }
        )

    summary = {
        "total_windows": len(window_audits),
        "windows_with_new_strikes": windows_with_new,
        "windows_without_new_strikes": windows_without_new,
        "windows_with_added_pcs": windows_with_added,
        "no_new_strike_reason_counts": dict(sorted(reason_counts.items())),
    }
    return window_audits, summary


def _finalize_variant_score(
    source_score: ScoreIR,
    variant_name: str,
    melody_events: list[NoteEvent],
    acc_events: list[NoteEvent],
    scale: F,
    transpose: int,
    scaled_measures: list[MeasureSpan],
    scaled_tempos: list[TempoEvent],
    scaled_total_q: F,
    mel_adj_records: list[dict[str, Any]],
    acc_stats: dict[str, Any],
    source_melody_count: int,
    source_accomp_pool_count: int,
) -> tuple[ScoreIR, dict[str, Any]]:
    """Combine melody & accompaniment, merge simultaneous same-pitch events, clamp overlapping durations."""
    # Deduplicate simultaneous (midi, onset_q) events if any arose, keeping melody role precedence.
    grouped: dict[tuple[int, F], list[NoteEvent]] = defaultdict(list)
    for ev in melody_events + acc_events:
        if ev.midi not in ALLOWED_SET:
            raise ValueError(f"Generated illegal MIDI pitch {ev.midi}")
        grouped[(ev.midi, ev.onset_q)].append(ev)

    merged_events: list[NoteEvent] = []
    dedup_merged = 0
    for idx, ((midi_val, onset_q), grp) in enumerate(sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])), start=1):
        if len(grp) > 1:
            dedup_merged += len(grp) - 1
        mel_items = [g for g in grp if g.role == "melody"]
        chosen = mel_items[0] if mel_items else grp[0]
        merged_refs: list[SourceRef] = []
        for g in grp:
            merged_refs.extend(g.source_refs)
        max_dur = max(g.duration_q for g in grp)
        max_dur = min(max_dur, scaled_total_q - onset_q)
        if max_dur <= 0:
            max_dur = F(1, 4)
        merged_events.append(
            NoteEvent(
                id=f"{variant_name}_{idx:05d}",
                midi=midi_val,
                onset_q=onset_q,
                duration_q=max_dur,
                role=chosen.role,
                source_refs=merged_refs,
                part="P1" if chosen.role == "melody" else "P2",
                voice="1",
                staff="1",
                original_midi=chosen.original_midi,
                octave_adjusted=chosen.octave_adjusted,
                adjustment_reason=chosen.adjustment_reason,
            )
        )

    # Clamp durations cleanly before the next strike of the SAME pitch and verify >=1q gap invariant.
    by_midi: dict[int, list[NoteEvent]] = defaultdict(list)
    for ev in merged_events:
        by_midi[ev.midi].append(ev)

    clamped_records: list[dict[str, Any]] = []
    for midi_val, evs in by_midi.items():
        evs.sort(key=lambda e: e.onset_q)
        for i in range(len(evs) - 1):
            cur = evs[i]
            nxt = evs[i + 1]
            gap = nxt.onset_q - cur.onset_q
            if gap < F(1):
                raise AssertionError(f"Variant {variant_name} violated >=1q gap on MIDI {midi_val}: gap={gap}")
            if cur.onset_q + cur.duration_q > nxt.onset_q:
                old_dur = cur.duration_q
                cur.duration_clamped_from_q = old_dur
                cur.duration_q = gap
                clamped_records.append(
                    {
                        "event_id": cur.id,
                        "role": cur.role,
                        "midi": midi_val,
                        "pitch": pitch_name(midi_val),
                        "onset_q": str(cur.onset_q),
                        "next_same_pitch_event_id": nxt.id,
                        "next_same_pitch_role": nxt.role,
                        "next_same_pitch_onset_q": str(nxt.onset_q),
                        "clamped_from_q": str(old_dur),
                        "clamped_to_q": str(cur.duration_q),
                    }
                )

    merged_events.sort(key=lambda e: (0 if e.part == "P1" else 1, e.onset_q, e.midi))
    window_audits, window_summary = _build_per_window_audit(
        final_events=merged_events,
        window_traces=acc_stats.get("window_traces", []),
        texture=variant_name,
    )
    public_acc_stats = {k: v for k, v in acc_stats.items() if k != "window_traces"}

    out_score = ScoreIR(
        parts=["P1", "P2"],
        part_names={"P1": "Melody", "P2": "Accompaniment"},
        events=merged_events,
        tempos=list(scaled_tempos),
        measures=list(scaled_measures),
        total_q=scaled_total_q,
        source_path=source_score.source_path,
        source_sha256=source_score.source_sha256,
        warnings=list(source_score.warnings),
        metadata={
            "variant": variant_name,
            "time_scale": str(scale),
            "transpose_semitones": transpose,
            "source_transpositions_stripped": True,
            "writing_concert_pitch_no_transpose_tag": True,
        },
    )

    change_log = {
        "variant": variant_name,
        "time_scale": str(scale),
        "transpose_semitones": transpose,
        "melody_strikes_source": source_melody_count,
        "melody_strikes_preserved": sum(1 for e in merged_events if e.role == "melody"),
        "melody_octave_adjusted_count": len(mel_adj_records),
        "melody_octave_adjustments": mel_adj_records,
        "accompaniment_source_pool_count": source_accomp_pool_count,
        "accompaniment_generated_count": sum(1 for e in merged_events if e.role == "accompaniment"),
        "accompaniment_revoiced_from_source": True,
        "accompaniment_source_pc_only": True,
        "accompaniment_stats": public_acc_stats,
        "harmony_window_summary": window_summary,
        "harmony_windows": window_audits,
        "simultaneous_same_pitch_merged": dedup_merged,
        "duration_clamped_count": len(clamped_records),
        "duration_clamps": clamped_records,
        "duration_clamps_sample": clamped_records[:25],
    }
    return out_score, serialize_rationals(change_log)


def arrange_score(score: ScoreIR, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Search feasible (time_scale, transpose) candidates and synthesize sparse/flowing ScoreIR variants."""
    opts = options or {}
    melody_part = opts.get("melody_part")
    texture = str(opts.get("texture", "both")).lower()
    if texture not in ("sparse", "flowing", "both"):
        raise UnsupportedScoreError(
            f"Unsupported texture '{texture}'. Expected sparse, flowing, or both.",
            issues=[{"code": "invalid_texture", "texture": texture}],
        )

    explicit_scale = parse_fraction(opts["time_scale"]) if opts.get("time_scale") is not None else None
    if explicit_scale is not None and explicit_scale <= 0:
        raise UnsupportedScoreError(
            f"Invalid --time-scale '{explicit_scale}': must be positive.",
            issues=[{"code": "invalid_time_scale", "time_scale": str(explicit_scale)}],
        )
    explicit_tr = int(opts["transpose"]) if opts.get("transpose") is not None else None
    beam_width = max(1, int(opts.get("beam_width", 64)))

    analysis, melody_strikes, accomp_pool, harmony_windows, chosen_part = _decompose_score(
        score, melody_part=melody_part
    )

    scales = (explicit_scale,) if explicit_scale is not None else DEFAULT_SCALES
    transposes = (explicit_tr,) if explicit_tr is not None else DEFAULT_TRANSPOSES

    candidates: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_sort_key: tuple[Any, ...] | None = None

    min_src_bpm = min(t.quarter_bpm for t in score.tempos)
    max_src_bpm = max(t.quarter_bpm for t in score.tempos)

    for scale in scales:
        scaled_min_bpm = min_src_bpm * scale
        scaled_max_bpm = max_src_bpm * scale
        bpm_ok = (scaled_min_bpm >= MIN_BPM) and (scaled_max_bpm <= MAX_BPM)

        scaled_sigs: list[tuple[int, int]] = []
        sig_ok = True
        for m in score.measures:
            sig = _scale_time_signature(m.beats, m.beat_type, m.nominal_q, scale)
            if sig is None:
                sig_ok = False
                break
            scaled_sigs.append(sig)

        for tr in transposes:
            if not bpm_ok:
                candidates.append(
                    {
                        "time_scale": str(scale),
                        "transpose_semitones": tr,
                        "scaled_bpm_range": [str(scaled_min_bpm), str(scaled_max_bpm)],
                        "feasible": False,
                        "status": "infeasible_bpm",
                        "reason": f"Scaled BPM range [{scaled_min_bpm}, {scaled_max_bpm}] outside [{MIN_BPM}, {MAX_BPM}].",
                    }
                )
                continue
            if not sig_ok:
                candidates.append(
                    {
                        "time_scale": str(scale),
                        "transpose_semitones": tr,
                        "scaled_bpm_range": [str(scaled_min_bpm), str(scaled_max_bpm)],
                        "feasible": False,
                        "status": "infeasible_time_signature",
                        "reason": f"Cannot cleanly represent scaled measure meter at time_scale={scale}.",
                    }
                )
                continue

            search_res = _beam_search_melody(
                melody_strikes=melody_strikes,
                scale=scale,
                transpose=tr,
                measures=score.measures,
                beam_width=beam_width,
            )
            if not search_res["feasible"]:
                candidates.append(
                    {
                        "time_scale": str(scale),
                        "transpose_semitones": tr,
                        "scaled_bpm_range": [str(scaled_min_bpm), str(scaled_max_bpm)],
                        "feasible": False,
                        "status": "infeasible_melody_conflict",
                        "failed_step_index": search_res["failed_step_index"],
                        "conflict_location": search_res["conflict_location"],
                    }
                )
                continue

            cand_entry = {
                "time_scale": str(scale),
                "transpose_semitones": tr,
                "scaled_bpm_range": [str(scaled_min_bpm), str(scaled_max_bpm)],
                "feasible": True,
                "status": "feasible",
                "melody_cost": search_res["cost"],
                "melody_octave_adjusted_count": search_res["octave_adjusted_count"],
                "melody_strike_count": len(melody_strikes),
            }
            candidates.append(cand_entry)

            sort_key = (
                search_res["cost"],
                search_res["octave_adjusted_count"],
                abs(tr),
                abs(scale - F(1)),
                tr,
            )
            if best_sort_key is None or sort_key < best_sort_key:
                best_sort_key = sort_key
                best_payload = {
                    "scale": scale,
                    "transpose": tr,
                    "scaled_sigs": scaled_sigs,
                    "search_res": search_res,
                    "candidate": cand_entry,
                }

    candidates_summary = {
        "melody_part": chosen_part,
        "searched_count": len(candidates),
        "feasible_count": sum(1 for c in candidates if c["feasible"]),
        "selected": best_payload["candidate"] if best_payload else None,
        "candidates": serialize_rationals(candidates),
    }

    if best_payload is None:
        return {
            "status": "no_feasible_candidate",
            "analysis": analysis,
            "candidates": candidates_summary,
            "variants": {},
            "changes": {
                "status": "no_feasible_candidate",
                "reason": "No (time_scale, transpose) satisfied BPM [100,220], 30 pitches, and >=1q melody spacing.",
                "conflict_locations": [
                    c["conflict_location"] for c in candidates if "conflict_location" in c
                ][:10],
                "conflicts": [c for c in candidates if not c["feasible"]],
            },
        }

    chosen_scale: F = best_payload["scale"]
    chosen_tr: int = best_payload["transpose"]
    scaled_sigs: list[tuple[int, int]] = best_payload["scaled_sigs"]
    assigned_pitches: list[int] = best_payload["search_res"]["assigned_pitches"]

    scaled_total_q = score.total_q * chosen_scale
    scaled_measures = [
        MeasureSpan(
            measure_index=m.measure_index,
            measure=m.measure,
            start_q=m.start_q * chosen_scale,
            end_q=m.end_q * chosen_scale,
            length_q=m.length_q * chosen_scale,
            nominal_q=m.nominal_q * chosen_scale,
            beats=sig[0],
            beat_type=sig[1],
            implicit=m.implicit,
        )
        for m, sig in zip(score.measures, scaled_sigs)
    ]
    scaled_tempos = [
        TempoEvent(
            onset_q=t.onset_q * chosen_scale,
            quarter_bpm=t.quarter_bpm * chosen_scale,
            source=t.source,
        )
        for t in score.tempos
    ]

    mel_events, mel_adj_records = _build_melody_events(
        melody_strikes=melody_strikes,
        assigned_pitches=assigned_pitches,
        scale=chosen_scale,
        transpose=chosen_tr,
        scaled_total_q=scaled_total_q,
    )

    target_textures = ("sparse", "flowing") if texture == "both" else (texture,)
    variants: dict[str, ScoreIR] = {}
    variant_changes: dict[str, Any] = {}
    for tex in target_textures:
        acc_events, acc_stats = _generate_accompaniment(
            texture=tex,
            target_measures=scaled_measures,
            harmony_windows=harmony_windows,
            accomp_pool=accomp_pool,
            melody_events=mel_events,
            scale=chosen_scale,
            transpose=chosen_tr,
            scaled_total_q=scaled_total_q,
        )
        v_score, v_changes = _finalize_variant_score(
            source_score=score,
            variant_name=tex,
            melody_events=mel_events,
            acc_events=acc_events,
            scale=chosen_scale,
            transpose=chosen_tr,
            scaled_measures=scaled_measures,
            scaled_tempos=scaled_tempos,
            scaled_total_q=scaled_total_q,
            mel_adj_records=mel_adj_records,
            acc_stats=acc_stats,
            source_melody_count=len(melody_strikes),
            source_accomp_pool_count=len(accomp_pool),
        )
        variants[tex] = v_score
        variant_changes[tex] = v_changes

    return {
        "status": "ok",
        "analysis": analysis,
        "candidates": candidates_summary,
        "variants": variants,
        "changes": serialize_rationals(
            {
                "status": "ok",
                "selected_candidate": best_payload["candidate"],
                "variants": variant_changes,
            }
        ),
    }
