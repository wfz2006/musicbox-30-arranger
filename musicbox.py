#!/usr/bin/env python3
"""30-note music box arranger CLI (MVP-1)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import json  # noqa: E402

from inspect_notation_midi import _paths_refer_to_same_file  # noqa: E402
try:
    from musicbox_core import (  # noqa: E402
        UnsupportedScoreError,
        analyze_score,
        arrange_score,
        export_variant,
        load_score,
        serialize_rationals,
        validate_variant,
    )
    from musicbox_core.model import parse_fraction  # noqa: E402
    _IMPORT_ERROR: BaseException | None = None
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(serialize_rationals(data), ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _validate_io_paths(input_path: Path, out_dir: Path) -> str | None:
    """Return error string if input is missing or output path is unsafe/non-empty."""
    if not input_path.exists():
        return f"Input file does not exist: {input_path}"
    if not input_path.is_file():
        return f"Input path is not a file: {input_path}"
    if _paths_refer_to_same_file(out_dir, input_path):
        return f"Refusing --out '{out_dir}': collides with --input '{input_path}'."
    if out_dir.exists():
        if not out_dir.is_dir():
            return f"Refusing --out '{out_dir}': exists as a file, not a directory."
        if any(out_dir.iterdir()):
            return f"Refusing non-empty output directory '{out_dir}'. Please choose a fresh directory."
    try:
        resolved_in = input_path.resolve(strict=False)
        resolved_out = out_dir.resolve(strict=False)
        if resolved_in == resolved_out or resolved_out in resolved_in.parents:
            # Technically input lives inside out_dir only if out_dir contains input_path, which shouldn't overwrite.
            pass
    except OSError:
        pass
    return None


def _build_output_readme(
    input_path: Path,
    result: dict[str, Any],
    validations: dict[str, Any],
    output_files: dict[str, str],
) -> str:
    analysis = result["analysis"]
    cands = result["candidates"]
    sel = cands.get("selected")
    lines = [
        f"# 30-Note Music Box Arrangement Output (`{input_path.name}`)",
        "",
        "## Input & Analysis",
        f"- Source: `{input_path}`",
        f"- Source SHA-256: `{analysis.get('source_sha256')}`",
        f"- Selected Melody Part: `{analysis.get('selected_melody_part')}` ({analysis.get('melody_selection_method')})",
        f"- Melody Strikes: `{analysis['melody']['strike_count']}` ({analysis['melody']['pitch_range']})",
        f"- Source Total Length: `{analysis.get('total_q')}q` across `{analysis.get('measure_count')}` measures",
        "",
        "## Candidate & Validation Summary",
    ]
    if sel is None:
        lines.extend(
            [
                "- Status: **NO FEASIBLE CANDIDATE**",
                "- No valid arrangement XML/MIDI generated.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Selected Parameterization: `time_scale={sel['time_scale']}`, `transpose={sel['transpose_semitones']:+d}`, `scaled_bpm={sel['scaled_bpm_range']}`",
                f"- Melody Octave Adjustments: `{sel['melody_octave_adjusted_count']}`",
                "",
                "| Variant | Layer 1 (Constraints) | Layer 2 (Fidelity) | Listening (Layer 3) | Duration (s) | Melody / Accomp |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for v_name, val in validations.items():
            l1 = val["layer1_constraints"]["status"]
            l2 = val["layer2_fidelity"]["status"]
            l3 = val["layer3_musical_summary"]["listening_evaluation"]
            st = val["layer3_musical_summary"].get("objective_stats", {})
            lines.append(
                f"| `{v_name}` | `{l1}` | `{l2}` | `{l3}` | `{st.get('duration_seconds')}` | `{st.get('melody_count')}` / `{st.get('accompaniment_count')}` |"
            )
    lines.extend(["", "## Files"])
    for key, fpath in output_files.items():
        lines.append(f"- `{key}`: `{fpath}`")
    return "\n".join(lines) + "\n"


def _cmd_analyze(args: argparse.Namespace) -> int:
    err = _validate_io_paths(args.input, args.out)
    if err is not None:
        print(json.dumps({"status": "error", "command": "analyze", "error": err}, ensure_ascii=False, indent=2))
        return 2

    try:
        score = load_score(args.input)
        analysis = analyze_score(score, melody_part=args.melody_part)
    except UnsupportedScoreError as exc:
        print(
            json.dumps(
                {"status": "unsupported", "command": "analyze", **exc.to_dict()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    except (OSError, ET.ParseError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "command": "analyze", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    analysis_path = _write_json(args.out / "analysis.json", analysis)
    summary = {
        "status": "ok",
        "command": "analyze",
        "input": str(args.input),
        "outputs": {"analysis": str(analysis_path)},
        "diagnostics": {
            "selected_melody_part": analysis["selected_melody_part"],
            "melody_strike_count": analysis["melody"]["strike_count"],
            "harmony_windows": analysis["harmony_summary"]["window_count"],
            "warnings_count": len(analysis.get("warnings", [])),
        },
        "musical_changes": None,
    }
    print(json.dumps(serialize_rationals(summary), ensure_ascii=False, indent=2))
    return 0


def _cmd_arrange(args: argparse.Namespace) -> int:
    err = _validate_io_paths(args.input, args.out)
    if err is not None:
        print(json.dumps({"status": "error", "command": "arrange", "error": err}, ensure_ascii=False, indent=2))
        return 2

    options: dict[str, Any] = {
        "melody_part": args.melody_part,
        "texture": args.texture,
        "transpose": args.transpose,
    }
    if args.time_scale is not None:
        try:
            options["time_scale"] = parse_fraction(args.time_scale)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "command": "arrange",
                        "error": f"Invalid --time-scale '{args.time_scale}': {exc}",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2

    try:
        score = load_score(args.input)
        result = arrange_score(score, options=options)
    except UnsupportedScoreError as exc:
        print(
            json.dumps(
                {"status": "unsupported", "command": "arrange", **exc.to_dict()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    except (OSError, ET.ParseError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "command": "arrange", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    outputs["analysis"] = str(_write_json(args.out / "analysis.json", result["analysis"]))
    outputs["candidates"] = str(_write_json(args.out / "candidates.json", result["candidates"]))
    outputs["changes"] = str(_write_json(args.out / "changes.json", result["changes"]))

    if result["status"] != "ok":
        val_payload = {
            "status": "fail",
            "passed": False,
            "reason": "no_feasible_candidate",
            "variants": {},
        }
        outputs["validation"] = str(_write_json(args.out / "validation.json", val_payload))
        readme_path = args.out / "README.md"
        readme_path.write_text(_build_output_readme(args.input, result, {}, outputs), encoding="utf-8")
        outputs["readme"] = str(readme_path)
        summary = {
            "status": "no_feasible_candidate",
            "command": "arrange",
            "input": str(args.input),
            "outputs": outputs,
            "diagnostics": {
                "feasible_candidates": result["candidates"]["feasible_count"],
                "validation_status": "fail",
            },
            "musical_changes": result["changes"],
        }
        print(json.dumps(serialize_rationals(summary), ensure_ascii=False, indent=2))
        return 1

    validations: dict[str, Any] = {}
    all_passed = True
    staged_paths: dict[str, dict[str, Path]] = {}

    for v_name, v_score in result["variants"].items():
        v_paths = export_variant(v_score, args.out, v_name)
        v_val = validate_variant(v_score, v_paths)
        validations[v_name] = v_val
        staged_paths[v_name] = v_paths
        if not v_val["passed"]:
            all_passed = False

    if not all_passed:
        # Remove failed XML/MIDI artifacts so validation failure never leaves unverified score outputs on disk.
        for v_name, v_paths in staged_paths.items():
            if not validations[v_name]["passed"]:
                for fpath in v_paths.values():
                    if fpath.exists():
                        fpath.unlink()
    else:
        for v_name, v_paths in staged_paths.items():
            outputs[f"{v_name}_musicxml"] = str(v_paths["musicxml"])
            outputs[f"{v_name}_midi"] = str(v_paths["midi"])

    events_payload = {
        "selected_candidate": result["candidates"]["selected"],
        "variants": {v_name: v_score.to_dict() for v_name, v_score in result["variants"].items()},
    }
    outputs["arrangement_events"] = str(_write_json(args.out / "arrangement-events.json", events_payload))

    validation_payload = {
        "status": "pass" if all_passed else "fail",
        "passed": all_passed,
        "variants": validations,
    }
    outputs["validation"] = str(_write_json(args.out / "validation.json", validation_payload))

    readme_path = args.out / "README.md"
    readme_path.write_text(_build_output_readme(args.input, result, validations, outputs), encoding="utf-8")
    outputs["readme"] = str(readme_path)

    diag = {
        "selected_candidate": result["candidates"]["selected"],
        "validation_status": validation_payload["status"],
        "variants": {
            k: {
                "status": v["status"],
                "layer1": v["layer1_constraints"]["status"],
                "layer2": v["layer2_fidelity"]["status"],
                "listening": v["layer3_musical_summary"]["listening_evaluation"],
            }
            for k, v in validations.items()
        },
    }
    summary = {
        "status": "ok" if all_passed else "validation_failed",
        "command": "arrange",
        "input": str(args.input),
        "outputs": outputs,
        "diagnostics": diag,
        "musical_changes": result["changes"],
    }
    print(json.dumps(serialize_rationals(summary), ensure_ascii=False, indent=2))
    return 0 if all_passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="30-note music box MVP-1 analyzer and arranger.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Inspect score parts, melody strikes, tempos, and harmony windows.")
    p_an.add_argument("--input", type=Path, required=True, help="Source MusicXML (.musicxml/.xml/.mxl)")
    p_an.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    p_an.add_argument("--melody-part", type=str, default=None, help="Override automatic melody part ID")
    p_an.set_defaults(func=_cmd_analyze)

    p_ar = sub.add_parser("arrange", help="Arrange score into 30-note MusicXML + MIDI variants and validate.")
    p_ar.add_argument("--input", type=Path, required=True, help="Source MusicXML (.musicxml/.xml/.mxl)")
    p_ar.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    p_ar.add_argument("--melody-part", type=str, default=None, help="Override automatic melody part ID")
    p_ar.add_argument("--texture", choices=("sparse", "flowing", "both"), default="both")
    p_ar.add_argument("--transpose", type=int, default=None, help="Force semitone transposition")
    p_ar.add_argument("--time-scale", type=str, default=None, help="Force time scaling ratio (e.g. 1/2, 1, 2)")
    p_ar.set_defaults(func=_cmd_arrange)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if _IMPORT_ERROR is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "command": args.command,
                    "error": f"Missing required dependency: {_IMPORT_ERROR}. Install requirements.txt in your Python environment.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
