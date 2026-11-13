#!/usr/bin/env python3
"""Core regression and CLI end-to-end tests for 30-note music box MVP-1."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from musicbox_core import (  # noqa: E402
    ALLOWED_MIDI,
    MeasureSpan,
    NoteEvent,
    ScoreIR,
    TempoEvent,
    UnsupportedScoreError,
    analyze_score,
    arrange_score,
    export_variant,
    load_score,
    validate_variant,
)
from musicbox_core.exporter import encode_tempo_us  # noqa: E402

TMP_WORK_ROOT = PROJECT_ROOT / ".tmp_test_mvp"
CLI_SCRIPT = PROJECT_ROOT / "musicbox.py"


def _make_partwise_xml(
    parts_measures: dict[str, list[str]],
    part_names: dict[str, str] | None = None,
) -> str:
    """Build valid UTF-8 MusicXML 4.0 Partwise document from part -> measure XML bodies."""
    names = part_names or {pid: pid for pid in parts_measures}
    part_list = "\n".join(
        f'    <score-part id="{pid}"><part-name>{names.get(pid, pid)}</part-name></score-part>'
        for pid in parts_measures
    )
    parts_body = []
    for pid, measures_xml in parts_measures.items():
        parts_body.append(f'  <part id="{pid}">\n' + "\n".join(measures_xml) + "\n  </part>")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtd/partwise.dtd">\n'
        '<score-partwise version="4.0">\n'
        f"  <part-list>\n{part_list}\n  </part-list>\n"
        + "\n".join(parts_body)
        + "\n</score-partwise>\n"
    )


class TempWorkspaceMixin:
    def make_case_dir(self, prefix: str = "case") -> Path:
        TMP_WORK_ROOT.mkdir(parents=True, exist_ok=True)
        path = TMP_WORK_ROOT / f"{prefix}_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda p=path: shutil.rmtree(p, ignore_errors=True))
        return path


class TestImporter(TempWorkspaceMixin, unittest.TestCase):
    def test_cross_bar_ties_transpose_pickup_and_trailing_rest(self) -> None:
        xml_text = _make_partwise_xml(
            {
                "P1": [
                    # Measure 0: pickup 1/4 beat in 4/4 with tempo=120 and chromatic transpose=-2 (written D5 -> sounding C5=72)
                    (
                        '    <measure number="0" implicit="yes">\n'
                        '      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time>'
                        '<transpose><diatonic>-1</diatonic><chromatic>-2</chromatic></transpose></attributes>\n'
                        '      <direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome></direction-type><sound tempo="120"/></direction>\n'
                        '      <note><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration><voice>1</voice><type>quarter</type><tie type="start"/></note>\n'
                        '    </measure>'
                    ),
                    # Measure 1: tie stop half note (4q? no dur=4 -> 2q) + trailing half rest (dur=4 -> 2q), total bar=4q
                    (
                        '    <measure number="1">\n'
                        '      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>half</type><tie type="stop"/></note>\n'
                        '      <note><rest/><duration>4</duration><voice>1</voice><type>half</type></note>\n'
                        '    </measure>'
                    ),
                ]
            }
        )
        work = self.make_case_dir("imp_tie")
        xml_path = work / "tie_pickup.musicxml"
        xml_path.write_text(xml_text, encoding="utf-8")

        score = load_score(xml_path)
        self.assertEqual(score.total_q, F(5))
        self.assertEqual(len(score.measures), 2)
        self.assertTrue(score.measures[0].implicit)
        self.assertEqual(score.measures[0].length_q, F(1))
        self.assertEqual(score.measures[1].length_q, F(4))
        self.assertEqual(len(score.events), 1)
        ev = score.events[0]
        self.assertEqual(ev.midi, 72)  # Sounding C5, NOT written D5 (74)
        self.assertEqual(ev.onset_q, F(0))
        self.assertEqual(ev.duration_q, F(3))
        self.assertEqual(len(ev.source_refs), 2)
        self.assertEqual(len(score.metadata["source_transpositions"]), 1)
        self.assertEqual(score.metadata["source_transpositions"][0]["kind"], "transpose")
        self.assertIn("-2", str(score.metadata["source_transpositions"][0]))

    def test_unsupported_structures_localize_part_and_measure(self) -> None:
        base_head = (
            '    <measure number="1"><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>'
            '<direction><sound tempo="120"/></direction>'
            '<note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note></measure>'
        )
        cases = [
            (
                "repeat",
                base_head
                + '    <measure number="2"><barline location="right"><repeat direction="backward"/></barline>'
                '<note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note></measure>',
                "unexpanded_navigation",
            ),
            (
                "grace",
                base_head
                + '    <measure number="2"><note><grace/><pitch><step>E</step><octave>5</octave></pitch><voice>1</voice><type>eighth</type></note>'
                '<note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note></measure>',
                "grace",
            ),
            (
                "orphan_tie",
                base_head
                + '    <measure number="2"><note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><tie type="stop"/></note></measure>',
                "orphan_tie",
            ),
        ]
        work = self.make_case_dir("imp_err")
        for label, body, expected_code in cases:
            xml_path = work / f"{label}.musicxml"
            xml_path.write_text(_make_partwise_xml({"P1": [body]}), encoding="utf-8")
            with self.assertRaises(UnsupportedScoreError, msg=label) as ctx:
                load_score(xml_path)
            payload = ctx.exception.to_dict()
            self.assertEqual(payload["issues"][0]["part"], "P1")
            self.assertEqual(str(payload["issues"][0]["measure"]), "2")
            self.assertIn(expected_code, payload["issues"][0]["code"])


class TestExporterAndValidation(TempWorkspaceMixin, unittest.TestCase):
    def test_asymmetric_durations_cross_bar_ties_pickup_and_220bpm(self) -> None:
        # Exercise 220 BPM (written_us=272728 -> <= 220), pickup implicit bar, cross-bar tie,
        # asymmetric durations (5/4, 7/4, 5/2, 9/2, 6, 1/3 triplet), and trailing rest.
        written_us, tempo_meta = encode_tempo_us(F(220))
        self.assertEqual(written_us, 272728)
        self.assertLessEqual(F(60_000_000, written_us), F(220))
        self.assertLessEqual(abs(tempo_meta["us_delta"]), 1.0)

        measures = [
            MeasureSpan(measure_index=0, measure="0", start_q=F(0), end_q=F(1), length_q=F(1), nominal_q=F(4), beats=4, beat_type=4, implicit=True),
            MeasureSpan(measure_index=1, measure="1", start_q=F(1), end_q=F(5), length_q=F(4), nominal_q=F(4), beats=4, beat_type=4),
            MeasureSpan(measure_index=2, measure="2", start_q=F(5), end_q=F(13), length_q=F(8), nominal_q=F(8), beats=4, beat_type=2),
            MeasureSpan(measure_index=3, measure="3", start_q=F(13), end_q=F(21), length_q=F(8), nominal_q=F(8), beats=4, beat_type=2),
        ]
        events = [
            # Pickup 1q C5 tied across barline into bar 1 (total 7/4)
            NoteEvent(id="e1", midi=72, onset_q=F(0), duration_q=F(7, 4), role="melody", part="P1"),
            # Triplet 1/3 at q=2, plus 5/4 at q=3, plus 6q across bars 2->3
            NoteEvent(id="e2", midi=74, onset_q=F(2), duration_q=F(1, 3), role="melody", part="P1"),
            NoteEvent(id="e3", midi=76, onset_q=F(3), duration_q=F(5, 4), role="melody", part="P1"),
            NoteEvent(id="e4", midi=77, onset_q=F(5), duration_q=F(6), role="melody", part="P1"),
            # Accompaniment spans 5/2 and 9/2 + trailing 1q rest in bar 3 (20..21 empty!)
            NoteEvent(id="a1", midi=60, onset_q=F(1), duration_q=F(5, 2), role="accompaniment", part="P2"),
            NoteEvent(id="a2", midi=64, onset_q=F(4), duration_q=F(9, 2), role="accompaniment", part="P2"),
            NoteEvent(id="a3", midi=67, onset_q=F(13), duration_q=F(7), role="accompaniment", part="P2"),
        ]
        score = ScoreIR(
            parts=["P1", "P2"],
            part_names={"P1": "Melody", "P2": "Accompaniment"},
            events=events,
            tempos=[TempoEvent(onset_q=F(0), quarter_bpm=F(220))],
            measures=measures,
            total_q=F(21),
        )
        work = self.make_case_dir("exp_rt")
        paths = export_variant(score, work, "roundtrip_220")
        val = validate_variant(score, paths)
        self.assertTrue(val["passed"], json.dumps(val, indent=2, ensure_ascii=False))
        self.assertEqual(val["layer1_constraints"]["status"], "pass")
        self.assertEqual(val["layer2_fidelity"]["status"], "pass")
        self.assertEqual(val["layer2_fidelity"]["midi_vs_ir"]["expected_tempos"][0][1], 272728)
        self.assertEqual(val["layer3_musical_summary"]["listening_evaluation"], "not_evaluated")

        # Tamper XML pitch illegally -> validate_variant MUST reject!
        tampered_xml_dir = self.make_case_dir("tamper_xml")
        shutil.copy2(paths["midi"], tampered_xml_dir / "t.mid")
        t_xml = tampered_xml_dir / "t.musicxml"
        t_xml.write_text(
            paths["musicxml"].read_text(encoding="utf-8").replace("<octave>5</octave>", "<octave>7</octave>", 1),
            encoding="utf-8",
        )
        bad_xml_val = validate_variant(score, {"musicxml": t_xml, "midi": tampered_xml_dir / "t.mid"})
        self.assertFalse(bad_xml_val["passed"])

        # Tamper MIDI bytes (drop last byte or append extra note-on byte? Simply corrupt EOT track byte / drop note) -> MUST FAIL!
        tampered_mid_dir = self.make_case_dir("tamper_mid")
        shutil.copy2(paths["musicxml"], tampered_mid_dir / "t.musicxml")
        m_bytes = bytearray(paths["midi"].read_bytes())
        # Flip one pitch byte (72 -> 73) in the MIDI stream so XML vs MIDI + IR check fails!
        idx = m_bytes.find(bytes([0x90, 72]))
        self.assertGreater(idx, 0)
        m_bytes[idx + 1] = 73
        t_mid = tampered_mid_dir / "t.mid"
        t_mid.write_bytes(bytes(m_bytes))
        bad_mid_val = validate_variant(score, {"musicxml": tampered_mid_dir / "t.musicxml", "midi": t_mid})
        self.assertFalse(bad_mid_val["passed"])


class TestArrangeCore(TempWorkspaceMixin, unittest.TestCase):
    def test_melody_self_conflict_and_melody_priority_over_accompaniment(self) -> None:
        # P1 has rapid repeated F5 (midi=77) at q=0, 1/2, 1 + chromatic Eb5 (75) and G#4 (68, outside 30!).
        # P2 has clashing same-pitch C5 (72) right next to P1 C5 at q=2, testing melody priority.
        measures = [
            MeasureSpan(measure_index=0, measure="1", start_q=F(0), end_q=F(4), length_q=F(4), nominal_q=F(4), beats=4, beat_type=4),
            MeasureSpan(measure_index=1, measure="2", start_q=F(4), end_q=F(8), length_q=F(4), nominal_q=F(4), beats=4, beat_type=4),
        ]
        events = [
            NoteEvent(id="m1", midi=77, onset_q=F(0), duration_q=F(1, 2), role="source", part="P1"),
            NoteEvent(id="m2", midi=77, onset_q=F(1, 2), duration_q=F(1, 2), role="source", part="P1"),
            NoteEvent(id="m3", midi=77, onset_q=F(1), duration_q=F(1), role="source", part="P1"),
            NoteEvent(id="m4", midi=72, onset_q=F(2), duration_q=F(1), role="source", part="P1"),
            NoteEvent(id="m5", midi=68, onset_q=F(3), duration_q=F(1), role="source", part="P1"),  # G#4 not in ALLOWED_MIDI (only G#5=80)
            NoteEvent(id="m6", midi=75, onset_q=F(4), duration_q=F(2), role="source", part="P1"),
            # Clashing accompaniment right near P1 notes + long held harmony
            NoteEvent(id="p1", midi=72, onset_q=F(3, 2), duration_q=F(2), role="source", part="P2"),
            NoteEvent(id="p2", midi=53, onset_q=F(0), duration_q=F(4), role="source", part="P2"),
            NoteEvent(id="p3", midi=60, onset_q=F(4), duration_q=F(4), role="source", part="P2"),
        ]
        score = ScoreIR(
            parts=["P1", "P2"],
            part_names={"P1": "Melody", "P2": "Left"},
            events=events,
            tempos=[TempoEvent(onset_q=F(0), quarter_bpm=F(110))],
            measures=measures,
            total_q=F(8),
        )
        res = arrange_score(score, options={"texture": "both"})
        self.assertEqual(res["status"], "ok")
        sparse = res["variants"]["sparse"]
        flowing = res["variants"]["flowing"]

        # Verify ALL 6 melody strikes are preserved with legal pitches and >=1q per pitch across P1+P2!
        self.assertEqual(sum(1 for e in sparse.events if e.role == "melody"), 6)
        self.assertEqual(sum(1 for e in flowing.events if e.role == "melody"), 6)
        self.assertNotEqual(
            Counter((e.midi, e.onset_q) for e in sparse.events),
            Counter((e.midi, e.onset_q) for e in flowing.events),
        )
        self.assertGreater(
            sum(1 for e in flowing.events if e.role == "accompaniment"),
            sum(1 for e in sparse.events if e.role == "accompaniment"),
        )

        work = self.make_case_dir("arr_prio")
        for v_name, v_score in res["variants"].items():
            paths = export_variant(v_score, work, v_name)
            val = validate_variant(v_score, paths)
            self.assertTrue(val["passed"], f"{v_name}: {val['failure_reasons']}")

    def test_infeasible_rapid_single_octave_chromatic_returns_no_feasible_candidate(self) -> None:
        # F#5 (midi=78) and G#5 (midi=80) each only exist in 1 octave inside ALLOWED_MIDI, OR
        # at 120 BPM with 1/8q spacing: max scale=1 -> gap=1/8q (<1q); wait F#5 has ONLY 1 octave!
        # Wait: what if we transpose ALL 12 keys? To truly guarantee ALL 12 transpositions fail,
        # place 5 notes every 1/8q across 1/2q! Even with max scale=1 (since BPM=120 -> scale=2 gives 240 BPM > 220!),
        # max legal scale <= 1, and in 1/2q with max 4 octaves per pitch class (C only has 3 octaves: 60,72,84!),
        # 4 strikes of ANY single pitch within <1q exceeds max 3-octave cardinality!
        measures = [
            MeasureSpan(measure_index=0, measure="1", start_q=F(0), end_q=F(4), length_q=F(4), nominal_q=F(4), beats=4, beat_type=4),
        ]
        # 5 repetitions of the same pitch within [0, 3/4q) at 120 BPM (max scale=1 -> 120 BPM):
        # no pitch class in ALLOWED_MIDI has 5 distinct octaves (max is F/G with 4 octaves)!
        events = [
            NoteEvent(id=f"r{i}", midi=60, onset_q=F(i, 6), duration_q=F(1, 6), role="source", part="P1")
            for i in range(5)
        ]
        score = ScoreIR(
            parts=["P1"],
            part_names={"P1": "Solo"},
            events=events,
            tempos=[TempoEvent(onset_q=F(0), quarter_bpm=F(120))],
            measures=measures,
            total_q=F(4),
        )
        res = arrange_score(score)
        self.assertEqual(res["status"], "no_feasible_candidate")
        self.assertEqual(res["candidates"]["feasible_count"], 0)
        first_conflict = res["changes"]["conflict_locations"][0]
        self.assertEqual(first_conflict["part"], "P1")
        self.assertEqual(first_conflict["measure"], "1")

    def test_simultaneous_same_pitch_merge_clamping_and_upcoming_melody_priority(self) -> None:
        # P1 voice 1 + voice 2 strike C5 simultaneously at q=0 (dur=1 and dur=3 -> merged dur=3, clamped to 2 when C5 strikes at q=2).
        # P2 wants F# harmony overlapping upcoming P1 melody F#5 at q=1/2 (single-octave pitch 78 in box!).
        xml_text = _make_partwise_xml(
            {
                "P1": [
                    (
                        '    <measure number="1">\n'
                        '      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>\n'
                        '      <direction><sound tempo="120"/></direction>\n'
                        '      <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><voice>1</voice><type>quarter</type></note>\n'
                        '      <note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>1</duration><voice>1</voice><type>eighth</type></note>\n'
                        '      <note><rest/><duration>1</duration><voice>1</voice><type>eighth</type></note>\n'
                        '      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>half</type></note>\n'
                        '      <backup><duration>8</duration></backup>\n'
                        '      <note><pitch><step>C</step><octave>5</octave></pitch><duration>6</duration><voice>2</voice><type>half</type><dot/></note>\n'
                        '      <note><rest/><duration>2</duration><voice>2</voice><type>quarter</type></note>\n'
                        '    </measure>'
                    ),
                ],
                "P2": [
                    (
                        '    <measure number="1">\n'
                        '      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>\n'
                        '      <note><chord-helper/><pitch><step>D</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice><type>half</type></note>\n'
                        '      <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>half</type></note>\n'
                        '    </measure>'
                    ),
                ],
            }
        )
        # Clean stray tag name; use valid MusicXML note element.
        xml_text = xml_text.replace("<chord-helper/>", "")
        work = self.make_case_dir("clamping_prio")
        xml_path = work / "clamp.musicxml"
        xml_path.write_text(xml_text, encoding="utf-8")

        ir = load_score(xml_path)
        c5_first = [e for e in ir.events if e.part == "P1" and e.onset_q == F(0) and e.midi == 72]
        self.assertEqual(len(c5_first), 1)
        self.assertEqual(len(c5_first[0].source_refs), 2)
        self.assertEqual(c5_first[0].duration_q, F(3))

        # Force tr=0, scale=1 so C5@q=0 (dur 3 -> clamped to 2) and F#5@q=1/2 remain exact!
        res = arrange_score(ir, options={"texture": "flowing", "transpose": 0, "time_scale": "1"})
        self.assertEqual(res["status"], "ok")
        flowing = res["variants"]["flowing"]
        mel_events = [e for e in flowing.events if e.role == "melody"]
        self.assertEqual(len(mel_events), 3)
        c5_out = [e for e in mel_events if e.onset_q == F(0) and e.midi == 72][0]
        self.assertEqual(c5_out.duration_q, F(2))
        self.assertEqual(c5_out.duration_clamped_from_q, F(3))
        self.assertGreaterEqual(res["changes"]["variants"]["flowing"]["duration_clamped_count"], 1)
        # Upcoming F#5@q=1/2 must NOT be pre-empted by accompaniment F#5@q=0!
        acc_fsharp_near = [
            e for e in flowing.events
            if e.role == "accompaniment" and e.midi == 78 and abs(e.onset_q - F(1, 2)) < F(1)
        ]
        self.assertEqual(acc_fsharp_near, [])
        paths = export_variant(flowing, work / "out", "flowing")
        self.assertTrue(validate_variant(flowing, paths)["passed"])

    def test_accompaniment_time_scaling_applied_once_no_double_scale(self) -> None:
        # Exact equivalent of planner timing_probe.py: 2 measures (8q), 4 half-bar harmony windows,
        # sustained high C6 lead without conflict, tested across scale=1, scale=2, and scale=1/2.
        def build_two_bar_score(src_bpm: F) -> ScoreIR:
            chords = [(60, 64, 67), (62, 65, 69), (64, 67, 71), (65, 69, 72)]
            acc = [
                NoteEvent(f"h{w}_{p}", p, F(w * 2), F(2), "source", part="P2")
                for w, chord in enumerate(chords)
                for p in chord
            ]
            return ScoreIR(
                parts=["P1", "P2"],
                part_names={"P1": "Melody", "P2": "Harmony"},
                events=[NoteEvent("lead", 84, F(0), F(8), "source", part="P1")] + acc,
                tempos=[TempoEvent(F(0), src_bpm)],
                measures=[
                    MeasureSpan(i, str(i + 1), F(i * 4), F((i + 1) * 4), F(4), F(4))
                    for i in range(2)
                ],
                total_q=F(8),
            )

        # scale=1 (100 BPM) vs scale=2 (200 BPM): sparse slots scale from [0,2,4,6] to [0,4,8,12]
        score_100 = build_two_bar_score(F(100))
        res_1 = arrange_score(score_100, {"texture": "both", "melody_part": "P1", "transpose": 0, "time_scale": "1"})
        res_2 = arrange_score(score_100, {"texture": "both", "melody_part": "P1", "transpose": 0, "time_scale": "2"})
        sp1_onsets = sorted({e.onset_q for e in res_1["variants"]["sparse"].events if e.role == "accompaniment"})
        sp2_onsets = sorted({e.onset_q for e in res_2["variants"]["sparse"].events if e.role == "accompaniment"})
        self.assertEqual(sp1_onsets, [F(0), F(2), F(4), F(6)])
        self.assertEqual(sp2_onsets, [F(0), F(4), F(8), F(12)])

        # Flowing at scale=2 covers both bars [0, 8) and [8, 16) cleanly without dropping bar 2!
        fl2_acc = [e for e in res_2["variants"]["flowing"].events if e.role == "accompaniment"]
        self.assertTrue(any(e.onset_q >= F(8) for e in fl2_acc))
        self.assertTrue(all(F(0) <= e.onset_q < F(16) and (e.onset_q + e.duration_q) <= F(16) for e in fl2_acc))

        # scale=1/2 (200 BPM source -> 100 BPM target, 2q bars spanning [0,2) and [2,4))
        score_200 = build_two_bar_score(F(200))
        res_half = arrange_score(score_200, {"texture": "both", "melody_part": "P1", "transpose": 0, "time_scale": "1/2"})
        sp_half_onsets = sorted({e.onset_q for e in res_half["variants"]["sparse"].events if e.role == "accompaniment"})
        fl_half_acc = [e for e in res_half["variants"]["flowing"].events if e.role == "accompaniment"]
        self.assertEqual(sp_half_onsets, [F(0), F(1), F(2), F(3)])
        self.assertTrue(any(e.onset_q >= F(2) for e in fl_half_acc))
        self.assertTrue(all(F(0) <= e.onset_q < F(4) and (e.onset_q + e.duration_q) <= F(4) for e in fl_half_acc))

    def test_default_accompaniment_uses_only_source_pcs_and_logs_per_window_decisions(self) -> None:
        # Bar 1 H1 [0,2): P2 has ONLY E4(64), B3(59? -> octave 71 in box!), D4(62) => PCs {E, B, D} (NO G# or G!).
        # Meanwhile P1 sings G#5(80) and G5(79): neither template (E7/Em7) nor melody authorizes G#/G in accompaniment!
        # Bar 1 H2 [2,4): P2 rests completely (source empty window).
        # Bar 2 H1 [4,6): P2 has C4(60), E4(64), G4(67); P1 hits dense melody (C5/D5/E5/F5 at 1/2q) triggering density thinning on flowing.
        measures = [
            MeasureSpan(0, "1", F(0), F(4), F(4), F(4), beats=4, beat_type=4),
            MeasureSpan(1, "2", F(4), F(8), F(4), F(4), beats=4, beat_type=4),
        ]
        events = [
            # Melody: held G#5 over H1, then dense 8ths in Bar 2
            NoteEvent("m1", 80, F(0), F(1), "source", part="P1"),
            NoteEvent("m2", 79, F(1), F(3), "source", part="P1"),
            NoteEvent("m3", 72, F(4), F(1, 2), "source", part="P1"),
            NoteEvent("m4", 74, F(9, 2), F(1, 2), "source", part="P1"),
            NoteEvent("m5", 76, F(5), F(1, 2), "source", part="P1"),
            NoteEvent("m6", 77, F(11, 2), F(1, 2), "source", part="P1"),
            NoteEvent("m7", 79, F(6), F(2), "source", part="P1"),
            # Accompaniment in Bar 1 H1: only E4(64), B3(59, outside box -> shifts octave to B4=71/B5=83), D4(62) -> PCs {E, B, D}
            NoteEvent("a1", 64, F(0), F(2), "source", part="P2"),
            NoteEvent("a2", 59, F(0), F(2), "source", part="P2"),
            NoteEvent("a3", 62, F(0), F(2), "source", part="P2"),
            NoteEvent("a4", 60, F(4), F(4), "source", part="P2"),
            NoteEvent("a5", 64, F(4), F(4), "source", part="P2"),
            NoteEvent("a6", 67, F(4), F(4), "source", part="P2"),
        ]
        score = ScoreIR(
            parts=["P1", "P2"],
            part_names={"P1": "Melody", "P2": "Left"},
            events=events,
            tempos=[TempoEvent(F(0), F(100))],
            measures=measures,
            total_q=F(8),
        )
        res = arrange_score(score, {"texture": "both", "melody_part": "P1", "transpose": 0, "time_scale": "2"})
        self.assertEqual(res["status"], "ok")

        for tex in ("sparse", "flowing"):
            v_score = res["variants"][tex]
            v_changes = res["changes"]["variants"][tex]
            # Assert Bar 1 H1 target [0,4): ONLY pitch classes in {E(4), B(11), D(2)} appear! NEVER G#(8) or G(7)!
            bar1_h1_acc = [e for e in v_score.events if e.role == "accompaniment" and F(0) <= e.onset_q < F(4)]
            self.assertTrue(bar1_h1_acc, f"{tex} should place allowed E/B/D accompaniment in Bar 1 H1")
            for ev in bar1_h1_acc:
                self.assertIn(ev.midi % 12, {4, 11, 2}, f"{tex} invented non-source pc {ev.midi % 12}")
                self.assertIn(ev.midi, ALLOWED_MIDI)

            # Bar 1 H2 [4,8): empty source accompaniment -> ZERO new accompaniment strikes placed!
            bar1_h2_acc = [e for e in v_score.events if e.role == "accompaniment" and F(4) <= e.onset_q < F(8)]
            self.assertEqual(bar1_h2_acc, [], f"{tex} invented accompaniment in empty source window Bar 1 H2")

            # Assert per-window changes.json audit structure exists and accurately reports added_pcs == [] and real event IDs
            hw_logs = v_changes["harmony_windows"]
            self.assertEqual(len(hw_logs), 4)
            valid_event_ids = {e.id for e in v_score.events}
            for wlog in hw_logs:
                self.assertEqual(wlog["added_pcs"], [], f"{tex} logged added_pcs={wlog['added_pcs']}")
                for eid in wlog["output_event_ids"]:
                    self.assertIn(eid, valid_event_ids)
                for sid in wlog["sustained_into_window_event_ids"]:
                    self.assertIn(sid, valid_event_ids)
                    self.assertNotIn(sid, wlog["output_event_ids"])
            self.assertIn("source_empty_window", str(hw_logs[1]["no_new_strike_reason"]))
            self.assertEqual(v_changes["harmony_window_summary"]["windows_with_added_pcs"], 0)
            if tex == "flowing":
                self.assertTrue(
                    any(s["status"] == "skipped_melody_density" for s in hw_logs[2]["slot_decisions"]),
                    f"Expected melody density skip log in Bar 2 H1: {hw_logs[2]['slot_decisions']}",
                )
                # Ensure sustained notes from H1 (e.g., bass at q=0 dur=4 into H2 if overlapping, or Bar 2 H1->H2) stay distinct from new strikes.
                self.assertTrue(set(hw_logs[2]["sustained_into_window_event_ids"]).isdisjoint(set(hw_logs[2]["output_event_ids"])))


class TestCLIWorkflow(TempWorkspaceMixin, unittest.TestCase):
    def test_synthetic_three_quarter_score_cli_from_root_and_refuses_non_empty_out(self) -> None:
        # Independent 3/4 score with C4 bass, E4/G4 harmony, F4-A4-C5-D5-E5 melody,
        # dotted quarter, pickup measure, explicit transpose=-2, 108 BPM.
        xml_text = _make_partwise_xml(
            {
                "RH": [
                    (
                        '    <measure number="1">\n'
                        '      <attributes><divisions>2</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes>\n'
                        '      <direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>108</per-minute></metronome></direction-type><sound tempo="108"/></direction>\n'
                        '      <note><pitch><step>G</step><octave>4</octave></pitch><duration>3</duration><voice>1</voice><type>quarter</type><dot/></note>\n'
                        '      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>eighth</type></note>\n'
                        '      <note><pitch><step>B</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><type>quarter</type></note>\n'
                        '    </measure>'
                    ),
                    (
                        '    <measure number="2">\n'
                        '      <note><pitch><step>C</step><octave>5</octave></pitch><duration>6</duration><voice>1</voice><type>half</type><dot/></note>\n'
                        '    </measure>'
                    ),
                ],
                "LH": [
                    (
                        '    <measure number="1">\n'
                        '      <attributes><divisions>2</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes>\n'
                        '      <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><type>quarter</type></note>\n'
                        '      <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>half</type></note>\n'
                        '    </measure>'
                    ),
                    (
                        '    <measure number="2">\n'
                        '      <note><pitch><step>F</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice><type>half</type></note>\n'
                        '      <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><type>quarter</type></note>\n'
                        '    </measure>'
                    ),
                ],
            },
            part_names={"RH": "Right Hand", "LH": "Left Hand"},
        )
        work = self.make_case_dir("cli_e2e")
        in_xml = work / "waltz.musicxml"
        in_xml.write_text(xml_text, encoding="utf-8")

        out_an = work / "analysis_out"
        res_an = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(CLI_SCRIPT), "analyze", "--input", str(in_xml), "--out", str(out_an)],
            cwd="C:\\",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_an.returncode, 0, res_an.stdout + res_an.stderr)
        self.assertTrue((out_an / "analysis.json").is_file())

        out_ar = work / "arrange_out"
        res_ar = subprocess.run(
            [
                sys.executable,
                "-B",
                "-X",
                "utf8",
                str(CLI_SCRIPT),
                "arrange",
                "--input",
                str(in_xml),
                "--out",
                str(out_ar),
                "--texture",
                "both",
            ],
            cwd="C:\\",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_ar.returncode, 0, res_ar.stdout + res_ar.stderr)
        expected_files = [
            "analysis.json",
            "candidates.json",
            "sparse.musicxml",
            "sparse.mid",
            "flowing.musicxml",
            "flowing.mid",
            "arrangement-events.json",
            "validation.json",
            "changes.json",
            "README.md",
        ]
        for fname in expected_files:
            self.assertTrue((out_ar / fname).is_file(), f"Missing output file: {fname}")

        val_json = json.loads((out_ar / "validation.json").read_text(encoding="utf-8"))
        self.assertTrue(val_json["passed"])
        self.assertEqual(val_json["variants"]["sparse"]["layer3_musical_summary"]["listening_evaluation"], "not_evaluated")
        self.assertEqual(val_json["variants"]["flowing"]["layer3_musical_summary"]["listening_evaluation"], "not_evaluated")

        # Re-running against non-empty directory MUST refuse with exit code 2 without clobbering existing files.
        before_bytes = (out_ar / "validation.json").read_bytes()
        res_rerun = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(CLI_SCRIPT), "arrange", "--input", str(in_xml), "--out", str(out_ar)],
            cwd="C:\\",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_rerun.returncode, 2)
        self.assertEqual((out_ar / "validation.json").read_bytes(), before_bytes)

        # Missing input exits with 2
        res_missing = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(CLI_SCRIPT), "analyze", "--input", str(work / "nope.musicxml"), "--out", str(work / "m_out")],
            cwd="C:\\",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_missing.returncode, 2)

        # CLI exit 2 when --out collides directly with --input file
        in_bytes = in_xml.read_bytes()
        res_col = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(CLI_SCRIPT), "arrange", "--input", str(in_xml), "--out", str(in_xml)],
            cwd="C:\\",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_col.returncode, 2)
        self.assertEqual(in_xml.read_bytes(), in_bytes)

        # CLI --melody-part / --transpose / --time-scale override works cleanly
        out_custom = work / "custom_out"
        res_custom = subprocess.run(
            [
                sys.executable, "-B", "-X", "utf8", str(CLI_SCRIPT), "arrange",
                "--input", str(in_xml), "--out", str(out_custom),
                "--melody-part", "RH", "--texture", "sparse", "--transpose", "2", "--time-scale", "2",
            ],
            cwd="C:\\",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_custom.returncode, 0, res_custom.stdout + res_custom.stderr)
        cand_custom = json.loads((out_custom / "candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(cand_custom["selected"]["time_scale"], "2")
        self.assertEqual(cand_custom["selected"]["transpose_semitones"], 2)
        self.assertTrue((out_custom / "sparse.mid").is_file())
        self.assertFalse((out_custom / "flowing.mid").exists())

        # CLI exit 1 on infeasible candidate writes diagnostics cleanly without fake XML/MIDI
        inf_xml = work / "infeasible.musicxml"
        inf_notes = "\n".join(
            '      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><voice>1</voice><type>16th</type></note>'
            for _ in range(5)
        )
        inf_xml.write_text(
            _make_partwise_xml(
                {
                    "P1": [
                        '    <measure number="1">\n'
                        '      <attributes><divisions>6</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>\n'
                        '      <direction><sound tempo="120"/></direction>\n'
                        + inf_notes
                        + '\n      <note><rest/><duration>19</duration><voice>1</voice><type>half</type></note>\n'
                        '    </measure>'
                    ]
                }
            ),
            encoding="utf-8",
        )
        out_inf = work / "inf_out"
        res_inf = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(CLI_SCRIPT), "arrange", "--input", str(inf_xml), "--out", str(out_inf)],
            cwd="C:\\",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_inf.returncode, 1, res_inf.stdout + res_inf.stderr)
        self.assertTrue((out_inf / "candidates.json").is_file())
        self.assertTrue((out_inf / "validation.json").is_file())
        self.assertFalse((out_inf / "sparse.musicxml").exists())
        self.assertFalse((out_inf / "sparse.mid").exists())

    def test_export_and_validate_partial_measure_whole_rest_in_4_2(self) -> None:
        # Reproduce 4/2 chord (4q) + 4q trailing rest followed by measure 2 strike.
        # Previously music21 expanded <rest><type>whole</type> into an 8q full-measure rest,
        # shifting Measure 2 accompaniment strikes from q=8 to q=12 in Layer 2 cross-check.
        score = ScoreIR(
            parts=["P1", "P2"],
            part_names={"P1": "Melody", "P2": "Accompaniment"},
            measures=[
                MeasureSpan(0, "1", F(0), F(8), F(8), F(8), 4, 2, False),
                MeasureSpan(1, "2", F(8), F(16), F(8), F(8), 4, 2, False),
            ],
            tempos=[TempoEvent(F(0), F(160))],
            events=[
                NoteEvent("m1", 72, F(0), F(4), "melody", part="P1"),
                NoteEvent("m2", 74, F(8), F(4), "melody", part="P1"),
                NoteEvent("a1", 67, F(0), F(4), "accompaniment", part="P2"),
                NoteEvent("a2", 71, F(0), F(4), "accompaniment", part="P2"),
                NoteEvent("a3", 60, F(8), F(4), "accompaniment", part="P2"),
            ],
            total_q=F(16),
            source_path="synthetic_4_2.musicxml",
        )
        work = self.make_case_dir("rest_4_2")
        exported = export_variant(score, work, "partial_rest")
        val = validate_variant(score, exported)
        self.assertEqual(val["status"], "pass", val)
        self.assertTrue(val["layer2_fidelity"]["music21_cross_check"]["xml_equals_ir"], val["layer2_fidelity"])


if __name__ == "__main__":
    unittest.main()
