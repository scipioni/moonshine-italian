"""Phase gates and latches: spike verdict, smoke record, WER gates.

Records under results/:
  spike/verdict.json      — spike latch (gates any long training run)
  smoke/record.json       — smoke success latch (gates final-train)
  gates/<name>.json       — individual gate outcomes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, load_config


def _results(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return REPO_ROOT / cfg["paths"]["results"]


def spike_verdict(cfg: dict | None = None) -> dict | None:
    p = _results(cfg) / "spike" / "verdict.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def smoke_record(cfg: dict | None = None) -> dict | None:
    p = _results(cfg) / "smoke" / "record.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def require_spike_ok(cfg: dict | None = None) -> None:
    """Fallback latch: no long training without a passing spike verdict."""
    cfg = cfg or load_config()
    verdict = spike_verdict(cfg)
    if verdict is None:
        raise SystemExit(
            "No spike verdict found (results/spike/verdict.json).\n"
            "Run the spikes first:  task spike   (or "
            "uv run python -m moonshine_it.spike all)\n"
            "If the spike failed, select a fallback base_model.selected_base "
            "in config.yaml before training."
        )
    if not verdict.get("ok"):
        failed = [k for k, v in verdict.get("spikes", {}).items() if not v]
        base = cfg["base_model"]["selected_base"]
        if base == "streaming-small":
            raise SystemExit(
                f"Spike verdict is FAILED ({', '.join(failed)}). Refusing to train "
                "with selected_base='streaming-small'.\n"
                "Choose a fallback in config.yaml (base_model.selected_base):\n"
                "  - 'non_streaming_base'  (bring up pipeline on moonshine-base)\n"
                "  - 'lora'                (moonshine-voice LoRA adapter path)\n"
                "then re-run the spikes for the selected base."
            )


def require_smoke_ok(cfg: dict | None = None) -> None:
    """Smoke latch: final training requires a recorded smoke success."""
    cfg = cfg or load_config()
    record = smoke_record(cfg)
    if record is None or not record.get("ok"):
        raise SystemExit(
            "No successful smoke record found (results/smoke/record.json).\n"
            "The final multi-day run must never start against an unvalidated "
            "process. Run the smoke chain first:  task smoke"
        )


def check_wer_gate(
    name: str,
    measured_wer: float,
    *,
    baseline_wer: float | None = None,
    cfg: dict | None = None,
) -> dict:
    """Evaluate a configured WER gate; writes results/gates/<name>.json.

    A gate failure raises SystemExit (blocks downstream targets).
    """
    cfg = cfg or load_config()
    gate = cfg["evaluation"]["gates"][name]
    relative = gate.get("relative_to") == "baseline"
    if relative and baseline_wer is None:
        raise SystemExit(
            f"gate '{name}' is relative to baseline but no baseline WER was "
            "provided — run the baseline spike first."
        )
    limit = gate["max_wer"] * (baseline_wer / 100.0 if relative else 1.0)
    outcome = {
        "gate": name,
        "measured_wer": measured_wer,
        "limit_wer": round(limit * 100, 2) if relative else limit * 100,
        "relative": relative,
        "baseline_wer": baseline_wer,
        "passed": measured_wer <= limit * 100,
    }
    out = _results(cfg) / "gates"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(outcome, indent=2))
    if not outcome["passed"]:
        raise SystemExit(
            f"GATE FAILED [{name}]: measured WER {measured_wer:.2f}% > "
            f"allowed {outcome['limit_wer']:.2f}% "
            f"({'baseline ' + str(baseline_wer) + '% x ' + str(gate['max_wer']) if relative else 'absolute'}). "
            "Downstream phases are blocked."
        )
    print(f"gate [{name}] passed: WER {measured_wer:.2f}% <= {outcome['limit_wer']:.2f}%")
    return outcome


def gate_record(name: str, cfg: dict | None = None) -> dict | None:
    p = _results(cfg) / "gates" / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def require_gate_passed(name: str, cfg: dict | None = None) -> dict:
    """Downstream latch: refuse to run when a gate was exceeded (or never run).

    Reads the gate outcome written by check_wer_gate; a missing or failed
    record blocks the calling phase (e.g. export).
    """
    cfg = cfg or load_config()
    record = gate_record(name, cfg)
    if record is None:
        raise SystemExit(
            f"GATE REQUIRED [{name}]: no gate record found "
            f"(results/gates/{name}.json).\n"
            "Run the evaluation with the gate first:\n"
            "  uv run python -m moonshine_it.evaluate_cli --model <ckpt> "
            f"--mode {load_config()['evaluation']['gates'][name].get('mode', 'full')} --gate {name}\n"
            "Downstream phases are blocked until the gate passes."
        )
    if not record.get("passed"):
        raise SystemExit(
            f"GATE FAILED [{name}]: measured WER {record.get('measured_wer')}% > "
            f"allowed {record.get('limit_wer')}%. Downstream phases are blocked.\n"
            "Re-train or relax the gate in config.yaml, then re-run the evaluation."
        )
    return record


def record_phase(kind: str, payload: dict, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = _results(cfg) / kind
    out.mkdir(parents=True, exist_ok=True)
    path = out / "record.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="moonshine_it.gates")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("require-smoke", help="exit non-zero without a smoke record")
    rec = sub.add_parser("record-smoke", help="write results/smoke/record.json")
    rec.add_argument("--phase", action="append", default=[],
                     help="phase name to record (repeatable)")
    rec.add_argument("--note", default="")
    args = parser.parse_args(argv)
    if args.cmd == "require-smoke":
        require_smoke_ok()
        print("smoke record: OK")
        return 0
    payload = {
        "ok": True,
        "phases": args.phase,
        "note": args.note,
        "recorded_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = record_phase("smoke", payload)
    print(f"smoke record written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
