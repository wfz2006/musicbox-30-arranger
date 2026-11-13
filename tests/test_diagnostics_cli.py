"""CLI tests for tools/inspect_notation_midi.py (R3 output collision guard)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_CLI = PROJECT_ROOT / "tools" / "inspect_notation_midi.py"
TEST_TMP_BASE = PROJECT_ROOT / ".tmp_cli_tests"


def _local_tempdir() -> tempfile.TemporaryDirectory[str]:
    TEST_TMP_BASE.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_TMP_BASE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_xml() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<score-partwise version="4.0">'
        '<part-list><score-part id="P1"><part-name>Box</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1">'
        "<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
        "<note><pitch><step>C</step><octave>5</octave></pitch>"
        "<duration>16</duration><type>whole</type></note>"
        "</measure></part></score-partwise>"
    )


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _synthetic_smf() -> bytes:
    events = bytearray()
    events.extend(_vlq(0) + b"\xFF\x51\x03\x07\xA1\x20")
    events.extend(_vlq(0) + b"\x90\x48\x5A")
    events.extend(_vlq(1920) + b"\x80\x48\x00")
    events.extend(_vlq(0) + b"\xFF\x2F\x00")
    return (
        b"MThd"
        + struct.pack(">IHHH", 6, 1, 1, 480)
        + b"MTrk"
        + struct.pack(">I", len(events))
        + bytes(events)
    )


class InspectNotationMidiCliTests(unittest.TestCase):
    def _run_cli(self, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(DIAG_CLI), *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def test_rejects_json_identical_to_xml_path(self) -> None:
        with _local_tempdir() as td:
            root = Path(td)
            xml_path = root / "synthetic.musicxml"
            xml_path.write_text(_synthetic_xml(), encoding="utf-8")
            before_bytes = xml_path.read_bytes()
            before_hash = _sha256(xml_path)

            proc = self._run_cli(["--xml", str(xml_path), "--json", str(xml_path)], cwd=root)
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertEqual(xml_path.read_bytes(), before_bytes)
            self.assertEqual(_sha256(xml_path), before_hash)
            self.assertIn(xml_path.name, proc.stderr)

    def test_rejects_json_colliding_with_midi_via_relative_and_absolute_paths(self) -> None:
        with _local_tempdir() as td:
            root = Path(td)
            sub = root / "nested"
            sub.mkdir()
            xml_path = root / "score.musicxml"
            midi_path = sub / "song.mid"
            xml_path.write_text(_synthetic_xml(), encoding="utf-8")
            midi_path.write_bytes(_synthetic_smf())
            xml_hash = _sha256(xml_path)
            midi_bytes = midi_path.read_bytes()
            midi_hash = _sha256(midi_path)

            proc = self._run_cli(
                ["--xml", "../score.musicxml", "--midi", "song.mid", "--json", str(midi_path.resolve())],
                cwd=sub,
            )
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertEqual(_sha256(xml_path), xml_hash)
            self.assertEqual(midi_path.read_bytes(), midi_bytes)
            self.assertEqual(_sha256(midi_path), midi_hash)
            self.assertIn("song.mid", proc.stderr)

    def test_rejects_json_colliding_with_xml_different_windows_case(self) -> None:
        with _local_tempdir() as td:
            root = Path(td)
            xml_path = root / "MixedCaseScore.musicxml"
            xml_path.write_text(_synthetic_xml(), encoding="utf-8")
            before_hash = _sha256(xml_path)
            upper_target = root / "MIXEDCASESCORE.MUSICXML"

            proc = self._run_cli(["--xml", str(xml_path), "--json", str(upper_target)], cwd=root)
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertEqual(_sha256(xml_path), before_hash)
            self.assertTrue(xml_path.read_text(encoding="utf-8").startswith("<?xml"))

    def test_rejects_json_aliasing_input_via_hardlink(self) -> None:
        with _local_tempdir() as td:
            root = Path(td)
            xml_path = root / "original.musicxml"
            alias_path = root / "hardlink_alias.musicxml"
            xml_path.write_text(_synthetic_xml(), encoding="utf-8")
            try:
                os.link(xml_path, alias_path)
            except OSError as exc:
                self.skipTest(f"Hardlinks unsupported in test dir: {exc}")

            before_hash = _sha256(xml_path)
            proc = self._run_cli(["--xml", str(xml_path), "--json", str(alias_path)], cwd=root)
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertEqual(_sha256(xml_path), before_hash)
            self.assertEqual(_sha256(alias_path), before_hash)
            self.assertIn("hardlink_alias.musicxml", proc.stderr)

    def test_valid_distinct_json_output_preserves_inputs(self) -> None:
        with _local_tempdir() as td:
            root = Path(td)
            xml_path = root / "score.musicxml"
            midi_path = root / "song.mid"
            out_path = root / "out" / "diagnostic.json"
            xml_path.write_text(_synthetic_xml(), encoding="utf-8")
            midi_path.write_bytes(_synthetic_smf())
            xml_hash = _sha256(xml_path)
            midi_hash = _sha256(midi_path)

            proc = self._run_cli(
                ["--xml", str(xml_path), "--midi", str(midi_path), "--json", str(out_path)],
                cwd=root,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr}")
            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["notation_durations"]["status"], "pass")
            self.assertEqual(payload["midi_onset_comparison"]["status"], "pass")
            self.assertEqual(_sha256(xml_path), xml_hash)
            self.assertEqual(_sha256(midi_path), midi_hash)


def tearDownModule() -> None:
    shutil.rmtree(TEST_TMP_BASE, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
