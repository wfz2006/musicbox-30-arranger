"""Unit tests for M0 notation duration and MIDI onset diagnostics."""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from inspect_notation_midi import (  # noqa: E402
    compare_xml_and_midi,
    extract_xml_strikes,
    inspect_midi_constraints,
    inspect_notation_durations,
    parse_midi_note_ons,
    summarize_xml_transposition,
)


def make_xml(measures_xml: str, part_id: str = "P1") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<score-partwise version="4.0">'
        f'<part-list><score-part id="{part_id}"><part-name>Box</part-name></score-part></part-list>'
        f'<part id="{part_id}">{measures_xml}</part>'
        "</score-partwise>"
    )


def make_measure(number: str, body: str, divisions: int | None = None, beats: int = 4, beat_type: int = 4) -> str:
    attrs = ""
    if divisions is not None:
        attrs = (
            f"<attributes><divisions>{divisions}</divisions>"
            f"<time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time></attributes>"
        )
    return f'<measure number="{number}">{attrs}{body}</measure>'


def make_note(
    step: str = "C",
    octave: int = 5,
    duration: int = 4,
    note_type: str | None = "quarter",
    dots: int = 0,
    alter: int = 0,
    ties: tuple[str, ...] = (),
    time_mod: tuple[int, int, str | None] | None = None,
    slurs: tuple[str, ...] = (),
) -> str:
    alt_xml = f"<alter>{alter}</alter>" if alter else ""
    tie_xml = "".join(f'<tie type="{t}"/>' for t in ties)
    type_xml = f"<type>{note_type}</type>" if note_type is not None else ""
    dot_xml = "<dot/>" * dots
    tm_xml = ""
    if time_mod is not None:
        act, norm, ntype = time_mod
        nt = f"<normal-type>{ntype}</normal-type>" if ntype else ""
        tm_xml = f"<time-modification><actual-notes>{act}</actual-notes><normal-notes>{norm}</normal-notes>{nt}</time-modification>"
    notations_inner = "".join(f'<tied type="{t}"/>' for t in ties) + "".join(
        f'<slur type="{s}" number="1"/>' for s in slurs
    )
    notations_xml = f"<notations>{notations_inner}</notations>" if notations_inner else ""
    return (
        f"<note><pitch><step>{step}</step>{alt_xml}<octave>{octave}</octave></pitch>"
        f"<duration>{duration}</duration>{tie_xml}{type_xml}{dot_xml}{tm_xml}{notations_xml}</note>"
    )


def vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def make_smf(ppq_or_division: int, events: list[tuple[int, bytes]], fmt: int = 1) -> bytes:
    """Build a minimal valid SMF byte payload from (abs_tick, raw_msg) events."""
    events_sorted = sorted(events, key=lambda item: item[0])
    track_data = bytearray()
    last_tick = 0
    for tick, msg in events_sorted:
        delta = tick - last_tick
        last_tick = tick
        track_data.extend(vlq(delta))
        track_data.extend(msg)
    track_data.extend(vlq(0))
    track_data.extend(b"\xFF\x2F\x00")
    header = b"MThd" + struct.pack(">IHHH", 6, fmt, 1, ppq_or_division)
    track = b"MTrk" + struct.pack(">I", len(track_data)) + bytes(track_data)
    return header + track


class NotationDurationTests(unittest.TestCase):
    def test_valid_dotted_quarter_matches_three_halves_q(self) -> None:
        xml = make_xml(
            make_measure(
                "1",
                make_note("C", 5, duration=6, note_type="quarter", dots=1)
                + make_note("D", 5, duration=2, note_type="eighth")
                + make_note("E", 5, duration=8, note_type="half"),
                divisions=4,
            )
        )
        res = inspect_notation_durations(xml, input_id="syn_dotted_ok")
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["mismatch_count"], 0)
        self.assertEqual(res["unsupported_count"], 0)
        self.assertEqual(res["consistent_count"], 3)

    def test_plain_quarter_with_nine_halves_actual_q_fails(self) -> None:
        xml = make_xml(
            make_measure(
                "1",
                make_note("E", 5, duration=36, note_type="quarter", dots=0),
                divisions=8,
                beats=4,
                beat_type=2,
            )
        )
        res = inspect_notation_durations(xml, input_id="syn_9_2_mismatch")
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["mismatch_count"], 1)
        m = res["mismatches"][0]
        self.assertEqual(m["part"], "P1")
        self.assertEqual(m["measure"], "1")
        self.assertEqual(m["measure_offset_q"], "0")
        self.assertEqual(m["abs_q"], "0")
        self.assertEqual(m["pitch"], "E5")
        self.assertEqual(m["divisions"], "8")
        self.assertEqual(m["duration_raw"], "36")
        self.assertEqual(m["actual_q"], "9/2")
        self.assertEqual(m["notated_q"], "1")
        self.assertEqual(m["type"], "quarter")
        self.assertEqual(m["dots"], 0)

    def test_eighth_triplets_match_one_third_q_each(self) -> None:
        triplet_note = make_note("C", 5, duration=1, note_type="eighth", time_mod=(3, 2, "eighth"))
        xml = make_xml(
            make_measure(
                "1",
                triplet_note * 3 + make_note("D", 5, duration=9, note_type="half", dots=1),
                divisions=3,
            )
        )
        res = inspect_notation_durations(xml, input_id="syn_triplet_ok")
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["mismatch_count"], 0)
        self.assertEqual(res["unsupported_count"], 0)
        self.assertEqual(res["consistent_count"], 4)

    def test_missing_type_is_unsupported_not_default_quarter(self) -> None:
        xml = make_xml(
            make_measure(
                "1",
                make_note("C", 5, duration=4, note_type=None)
                + make_note("D", 5, duration=12, note_type="half", dots=1),
                divisions=4,
            )
        )
        res = inspect_notation_durations(xml, input_id="syn_missing_type")
        self.assertEqual(res["status"], "unsupported")
        self.assertEqual(res["unsupported_count"], 1)
        self.assertEqual(res["mismatch_count"], 0)
        self.assertEqual(res["consistent_count"], 1)
        self.assertEqual(res["unsupported_items"][0]["reason"], "missing_type")

    def test_unknown_type_is_unsupported(self) -> None:
        xml = make_xml(
            make_measure(
                "1",
                make_note("C", 5, duration=4, note_type="weird-unit"),
                divisions=4,
            )
        )
        res = inspect_notation_durations(xml, input_id="syn_unknown_type")
        self.assertEqual(res["status"], "unsupported")
        self.assertEqual(res["unsupported_count"], 1)
        self.assertEqual(res["unsupported_items"][0]["reason"], "unknown_type")

    def test_whole_measure_rest_without_type_is_accepted_while_partial_untyped_rest_is_unsupported(self) -> None:
        xml_ok = make_xml(
            make_measure(
                "1",
                '<note><rest measure="yes"/><duration>12</duration></note>',
                divisions=4,
                beats=3,
                beat_type=4,
            )
        )
        res_ok = inspect_notation_durations(xml_ok, input_id="syn_measure_rest")
        self.assertEqual(res_ok["status"], "pass")
        self.assertEqual(res_ok["mismatch_count"], 0)
        self.assertEqual(res_ok["unsupported_count"], 0)
        self.assertEqual(res_ok["whole_measure_rest_count"], 1)

        xml_partial = make_xml(
            make_measure(
                "1",
                "<note><rest/><duration>4</duration></note>"
                + make_note("C", 5, duration=12, note_type="half", dots=1),
                divisions=4,
            )
        )
        res_partial = inspect_notation_durations(xml_partial, input_id="syn_partial_rest_no_type")
        self.assertEqual(res_partial["status"], "unsupported")
        self.assertEqual(res_partial["unsupported_count"], 1)

    def test_whole_measure_rest_with_wrong_duration_fails(self) -> None:
        xml_bad = make_xml(
            make_measure(
                "1",
                '<note><rest measure="yes"/><duration>4</duration></note>',
                divisions=4,
                beats=4,
                beat_type=4,
            )
        )
        res = inspect_notation_durations(xml_bad, input_id="syn_bad_measure_rest")
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["mismatch_count"], 1)
        self.assertEqual(res["mismatches"][0]["reason"], "whole_measure_rest_duration_mismatch")


class MidiOnsetComparisonTests(unittest.TestCase):
    def test_two_bar_tied_note_produces_single_strike_and_matches_midi(self) -> None:
        xml = make_xml(
            make_measure("1", make_note("C", 5, duration=16, note_type="whole", ties=("start",)), divisions=4)
            + make_measure("2", make_note("C", 5, duration=16, note_type="whole", ties=("stop",)))
        )
        # Note-on at tick 0 (C5=72, vel=90), velocity=0 note-off at tick 8*480=3840
        midi_bytes = make_smf(
            480,
            [
                (0, bytes([0x90, 72, 90])),
                (3840, bytes([0x90, 72, 0])),
            ],
        )
        strikes = extract_xml_strikes(xml)
        self.assertEqual(strikes["status"], "ok")
        self.assertEqual(strikes["noteheads"], 2)
        self.assertEqual(strikes["unique_strike_count"], 1)

        cmp_res = compare_xml_and_midi(xml, midi_bytes)
        self.assertEqual(cmp_res["status"], "pass")
        self.assertEqual(cmp_res["xml_unique_strikes"], 1)
        self.assertEqual(cmp_res["midi_real_note_ons"], 1)
        self.assertEqual(cmp_res["midi_zero_velocity_note_ons_ignored"], 1)
        self.assertEqual(cmp_res["extra_in_midi_count"], 0)
        self.assertEqual(cmp_res["missing_in_midi_count"], 0)

    def test_untied_repeated_note_produces_two_strikes(self) -> None:
        xml = make_xml(
            make_measure("1", make_note("C", 5, duration=16, note_type="whole"), divisions=4)
            + make_measure("2", make_note("C", 5, duration=16, note_type="whole"))
        )
        midi_two = make_smf(
            480,
            [
                (0, bytes([0x90, 72, 90])),
                (1920, bytes([0x80, 72, 0])),
                (1920, bytes([0x90, 72, 90])),
                (3840, bytes([0x80, 72, 0])),
            ],
        )
        cmp_two = compare_xml_and_midi(xml, midi_two)
        self.assertEqual(cmp_two["status"], "pass")
        self.assertEqual(cmp_two["xml_unique_strikes"], 2)
        self.assertEqual(cmp_two["midi_real_note_ons"], 2)

        midi_one = make_smf(480, [(0, bytes([0x90, 72, 90])), (3840, bytes([0x80, 72, 0]))])
        cmp_one = compare_xml_and_midi(xml, midi_one)
        self.assertEqual(cmp_one["status"], "fail")
        self.assertEqual(cmp_one["missing_in_midi_count"], 1)
        self.assertEqual(cmp_one["missing_in_midi"][0]["measure"], "2")
        self.assertEqual(cmp_one["missing_in_midi"][0]["abs_q"], "4")

    def test_extra_midi_note_on_at_tie_continuation_is_localized(self) -> None:
        xml = make_xml(
            make_measure("1", make_note("A", 5, duration=16, note_type="whole", ties=("start",)), divisions=4)
            + make_measure("2", make_note("A", 5, duration=16, note_type="whole", ties=("stop",)))
        )
        # Buggy MIDI re-strikes A5 (81) at bar 2 onset (abs_q=4 -> tick=1920)
        midi_buggy = make_smf(
            480,
            [
                (0, bytes([0x90, 81, 92])),
                (1920, bytes([0x80, 81, 0])),
                (1920, bytes([0x90, 81, 92])),
                (3840, bytes([0x80, 81, 0])),
            ],
        )
        cmp_res = compare_xml_and_midi(xml, midi_buggy)
        self.assertEqual(cmp_res["status"], "fail")
        self.assertEqual(cmp_res["xml_unique_strikes"], 1)
        self.assertEqual(cmp_res["midi_real_note_ons"], 2)
        self.assertEqual(cmp_res["extra_in_midi_count"], 1)
        extra = cmp_res["extra_in_midi"][0]
        self.assertEqual(extra["midi"], 81)
        self.assertEqual(extra["pitch"], "A5")
        self.assertEqual(extra["abs_q"], "4")
        self.assertEqual(extra["measure"], "2")
        self.assertEqual(extra["measure_offset_q"], "0")
        self.assertTrue(extra["matches_xml_tie_continuation"])

    def test_slur_does_not_merge_strikes(self) -> None:
        xml = make_xml(
            make_measure(
                "1",
                make_note("C", 5, duration=8, note_type="half", slurs=("start",))
                + make_note("C", 5, duration=8, note_type="half", slurs=("stop",)),
                divisions=4,
            )
        )
        strikes = extract_xml_strikes(xml)
        self.assertEqual(strikes["status"], "ok")
        self.assertEqual(strikes["unique_strike_count"], 2)

    def test_non_ppq_smpte_midi_reports_unsupported(self) -> None:
        # SMPTE division: bit 15 = 1, -25 fps (0xE7 = 231 -> signed -25), 40 ticks/frame (0x28)
        smpte_division = 0xE728
        midi_smpte = make_smf(smpte_division, [(0, bytes([0x90, 72, 90])), (100, bytes([0x80, 72, 0]))])
        parsed = parse_midi_note_ons(midi_smpte, input_id="syn_smpte")
        self.assertEqual(parsed["status"], "unsupported")
        self.assertIn("smpte", parsed["reason"])

        xml = make_xml(make_measure("1", make_note("C", 5, duration=16, note_type="whole"), divisions=4))
        cmp_res = compare_xml_and_midi(xml, midi_smpte)
        self.assertEqual(cmp_res["status"], "unsupported")

    def test_running_status_velocity_zero_handled(self) -> None:
        # Running status 0x90: note-on C5 vel 80, then delta 480 C5 vel 0 (note-off), then delta 0 D5 vel 80
        track_body = bytearray()
        track_body.extend(vlq(0) + bytes([0x90, 72, 80]))
        track_body.extend(vlq(480) + bytes([72, 0]))  # running status 0x90, vel=0 -> note off
        track_body.extend(vlq(0) + bytes([74, 80]))   # running status 0x90, vel=80 -> real note-on at 1q
        track_body.extend(vlq(480) + bytes([74, 0]))
        track_body.extend(vlq(0) + bytes([0xFF, 0x2F, 0x00]))
        midi_rs = (
            b"MThd"
            + struct.pack(">IHHH", 6, 1, 1, 480)
            + b"MTrk"
            + struct.pack(">I", len(track_body))
            + bytes(track_body)
        )
        parsed = parse_midi_note_ons(midi_rs, input_id="syn_running_status")
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["real_note_on_count"], 2)
        self.assertEqual(parsed["zero_velocity_note_ons_ignored"], 2)
        self.assertEqual([(e["midi"], str(e["abs_q"])) for e in parsed["note_ons"]], [(72, "0"), (74, "1")])

    def test_transposed_xml_matches_sounding_midi_pitch(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<score-partwise version="4.0">'
            '<part-list><score-part id="P1"><part-name>Box</part-name></score-part></part-list>'
            '<part id="P1"><measure number="1">'
            "<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<transpose><diatonic>0</diatonic><chromatic>0</chromatic><octave-change>2</octave-change></transpose>"
            "</attributes>"
            + make_note("C", 3, duration=16, note_type="whole")
            + "</measure></part></score-partwise>"
        )
        summary = summarize_xml_transposition(xml)
        self.assertEqual(summary["written_pitch_range"], ["C3", "C3"])
        self.assertEqual(summary["sounding_pitch_range"], ["C5", "C5"])
        self.assertEqual(summary["transpositions"], [{"part": "P1", "measure": "1", "staff": "all", "semitones": "24"}])

        midi_c5 = make_smf(480, [(0, bytes([0x90, 72, 90])), (1920, bytes([0x80, 72, 0]))])
        cmp_res = compare_xml_and_midi(xml, midi_c5)
        self.assertEqual(cmp_res["status"], "pass")


def make_multi_track_smf(
    ppq: int, track_events: list[list[tuple[int, bytes]]], fmt: int = 1
) -> bytes:
    """Build a multi-track SMF byte payload from per-track (abs_tick, raw_msg) lists."""
    chunks = bytearray()
    for events in track_events:
        track_data = bytearray()
        last_tick = 0
        for tick, msg in sorted(events, key=lambda item: item[0]):
            delta = tick - last_tick
            last_tick = tick
            track_data.extend(vlq(delta))
            track_data.extend(msg)
        track_data.extend(vlq(0))
        track_data.extend(b"\xFF\x2F\x00")
        chunks.extend(b"MTrk" + struct.pack(">I", len(track_data)) + bytes(track_data))
    header = b"MThd" + struct.pack(">IHHH", 6, fmt, len(track_events), ppq)
    return header + bytes(chunks)


def tempo_meta(us_per_quarter: int) -> bytes:
    """Encode an SMF Set Tempo (0x51) meta event."""
    return b"\xFF\x51\x03" + us_per_quarter.to_bytes(3, "big")


class MidiConstraintsTests(unittest.TestCase):
    def test_pitch_constraints_a3_fails_and_valid_boundary_pitches_pass(self) -> None:
        midi_bad = make_smf(
            480,
            [
                (0, tempo_meta(500_000)),
                (0, bytes([0x90, 57, 80])),  # A3: inside F3..A6 range but missing on 30-note tooth list
                (480, bytes([0x80, 57, 0])),
            ],
        )
        res_bad = inspect_midi_constraints(midi_bad, input_id="syn_a3")
        self.assertEqual(res_bad["status"], "fail")
        self.assertEqual(res_bad["pitch_violation_count"], 1)
        viol = res_bad["pitch_violations"][0]
        self.assertEqual((viol["track"], viol["tick"], str(viol["abs_q"]), viol["midi"], viol["pitch"]), (0, 0, "0", 57, "A3"))

        midi_ok = make_smf(
            480,
            [
                (0, tempo_meta(500_000)),
                (0, bytes([0x90, 53, 80])),    # F3
                (480, bytes([0x90, 70, 80])),  # Bb4
                (960, bytes([0x90, 75, 80])),  # Eb5
                (1440, bytes([0x90, 93, 80])), # A6
                (1920, bytes([0x80, 93, 0])),
            ],
        )
        res_ok = inspect_midi_constraints(midi_ok, input_id="syn_valid_range")
        self.assertEqual(res_ok["status"], "pass")
        self.assertEqual(res_ok["pitch_violation_count"], 0)
        self.assertEqual(res_ok["pitch_range"], ["F3", "A6"])

    def test_same_pitch_gap_and_distinct_pitch_gap(self) -> None:
        midi_1q = make_smf(
            480,
            [
                (0, tempo_meta(500_000)),
                (0, bytes([0x90, 72, 80])),
                (240, bytes([0x80, 72, 0])),
                (480, bytes([0x90, 72, 80])),  # 1q gap -> allowed
                (720, bytes([0x80, 72, 0])),
            ],
        )
        res_1q = inspect_midi_constraints(midi_1q, input_id="syn_1q")
        self.assertEqual(res_1q["status"], "pass")
        self.assertEqual(str(res_1q["minimum_same_pitch_gap_q"]), "1")
        self.assertEqual(res_1q["too_close_pair_count"], 0)

        midi_half_same = make_smf(
            480,
            [
                (0, tempo_meta(500_000)),
                (0, bytes([0x90, 72, 80])),
                (120, bytes([0x80, 72, 0])),
                (240, bytes([0x90, 72, 80])),  # 1/2q gap same pitch -> fail
                (480, bytes([0x80, 72, 0])),
            ],
        )
        res_half = inspect_midi_constraints(midi_half_same, input_id="syn_half_same")
        self.assertEqual(res_half["status"], "fail")
        self.assertEqual(str(res_half["minimum_same_pitch_gap_q"]), "1/2")
        self.assertEqual(res_half["too_close_pair_count"], 1)
        pair = res_half["too_close_pairs"][0]
        self.assertEqual(str(pair["gap_q"]), "1/2")
        self.assertEqual(pair["pitch"], "C5")

        midi_half_diff = make_smf(
            480,
            [
                (0, tempo_meta(500_000)),
                (0, bytes([0x90, 72, 80])),
                (240, bytes([0x90, 74, 80])),  # 1/2q gap different pitch -> pass
                (480, bytes([0x80, 74, 0])),
            ],
        )
        res_diff = inspect_midi_constraints(midi_half_diff, input_id="syn_half_diff")
        self.assertEqual(res_diff["status"], "pass")
        self.assertEqual(res_diff["too_close_pair_count"], 0)

    def test_tempo_boundaries_and_missing_timeline(self) -> None:
        for bpm_label, us_val in [("100", 600_000), ("220_max_int_us", 272_728)]:
            midi_ok = make_smf(480, [(0, tempo_meta(us_val)), (0, bytes([0x90, 72, 80])), (480, bytes([0x80, 72, 0]))])
            res = inspect_midi_constraints(midi_ok, input_id=f"syn_tempo_{bpm_label}")
            self.assertEqual(res["status"], "pass", msg=bpm_label)
            self.assertTrue(res["tempo_verified"])
            self.assertEqual(res["tempo_violation_count"], 0)

        for bpm_label, us_val in [("99", 606_061), ("221", 271_493)]:
            midi_bad = make_smf(480, [(0, tempo_meta(us_val)), (0, bytes([0x90, 72, 80])), (480, bytes([0x80, 72, 0]))])
            res = inspect_midi_constraints(midi_bad, input_id=f"syn_tempo_{bpm_label}")
            self.assertEqual(res["status"], "fail", msg=bpm_label)
            self.assertEqual(res["tempo_violation_count"], 1, msg=bpm_label)

        midi_no_tempo = make_smf(480, [(0, bytes([0x90, 72, 80])), (480, bytes([0x80, 72, 0]))])
        res_no_tempo = inspect_midi_constraints(midi_no_tempo, input_id="syn_no_tempo")
        self.assertEqual(res_no_tempo["status"], "fail")
        self.assertFalse(res_no_tempo["tempo_verified"])
        self.assertTrue(res_no_tempo["missing_initial_tempo"])

        midi_late_tempo = make_smf(
            480,
            [(0, bytes([0x90, 72, 80])), (960, tempo_meta(500_000)), (960, bytes([0x80, 72, 0]))],
        )
        res_late = inspect_midi_constraints(midi_late_tempo, input_id="syn_late_tempo")
        self.assertEqual(res_late["status"], "fail")
        self.assertFalse(res_late["tempo_verified"])
        self.assertTrue(res_late["missing_initial_tempo"])

        midi_zero_tempo = make_smf(
            480,
            [(0, b"\xFF\x51\x03\x00\x00\x00"), (0, bytes([0x90, 72, 80])), (480, bytes([0x80, 72, 0]))],
        )
        res_zero = inspect_midi_constraints(midi_zero_tempo, input_id="syn_zero_tempo")
        self.assertEqual(res_zero["status"], "unsupported")
        self.assertFalse(res_zero["tempo_verified"])

    def test_cross_track_gap_duplicates_and_velocity_zero_ignored(self) -> None:
        midi_multi = make_multi_track_smf(
            480,
            [
                [(0, tempo_meta(500_000)), (0, bytes([0x90, 72, 80])), (960, bytes([0x90, 76, 80]))],
                [
                    (240, bytes([0x91, 72, 75])),  # C5 1/2q after Track 0 C5 -> cross-track too close
                    (480, bytes([0x91, 72, 0])),   # velocity=0 note-off, must be ignored
                    (960, bytes([0x91, 76, 85])),  # E5 simultaneous with Track 0 E5 -> duplicate group
                    (1440, bytes([0x91, 76, 0])),
                ],
            ],
        )
        res = inspect_midi_constraints(midi_multi, input_id="syn_multitrack")
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["real_note_on_count"], 4)
        self.assertEqual(res["unique_strike_count"], 3)
        self.assertEqual(res["zero_velocity_note_ons_ignored"], 2)
        self.assertEqual(res["too_close_pair_count"], 1)
        self.assertEqual(res["duplicate_group_count"], 1)
        pair = res["too_close_pairs"][0]
        self.assertEqual((pair["previous"]["track"], pair["current"]["track"]), (0, 1))
        dup = res["duplicate_groups"][0]
        self.assertEqual((dup["pitch"], str(dup["abs_q"]), dup["count"]), ("E5", "2", 2))
        self.assertEqual({s["track"] for s in dup["strikes"]}, {0, 1})


if __name__ == "__main__":
    unittest.main()
