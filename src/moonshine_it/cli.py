"""moonshine-it CLI: dispatches subcommands to their modules."""

from __future__ import annotations

import sys

USAGE = """\
moonshine-it — Italian streaming Moonshine pipeline

usage: moonshine-it <command> [args]

commands:
  env-check    verify GPU torch/ORT, accelerator, profile
  download     model | data <fleurs|mls|common_voice>
  prepare      prepare a dataset (16 kHz, VAD segmentation, manifests)
  slice-smoke  deterministic smoke slice from prepared data
  spike        grad | tokenizer | baseline | verdict | all
  train        fine-tune (delegates to train.py args)
  eval         full/streaming WER/CER with optional gate
  export       ONNX export (gate-checked) + parity verification
  quantize     INT8 quantization + .ort serialization
  ort-eval     streaming eval of the .ort bundle + post-quant gate
  validate     validate + promote the .ort release bundle
  parity       ONNX-vs-PyTorch parity check
"""

COMMANDS = ("env-check", "download", "prepare", "slice-smoke", "spike",
            "train", "eval", "export", "quantize", "ort-eval", "validate",
            "parity")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0 if argv else 1
    cmd, rest = argv[0], argv[1:]

    if cmd == "env-check":
        from moonshine_it import env_check
        return env_check.main(rest)
    if cmd == "download":
        from moonshine_it import download
        return download.main(rest)
    if cmd == "prepare":
        from moonshine_it import prepare
        return prepare.main(rest)
    if cmd == "slice-smoke":
        from moonshine_it import slice_smoke
        return slice_smoke.main()
    if cmd == "spike":
        from moonshine_it import spike
        return spike.main(rest)
    if cmd == "train":
        from moonshine_it import train_loop
        return train_loop.main(rest)
    if cmd == "eval":
        from moonshine_it import evaluate_cli
        return evaluate_cli.main(rest)
    if cmd == "export":
        from moonshine_it import export
        return export.main(rest)
    if cmd == "quantize":
        from moonshine_it import quantize
        return quantize.main(rest)
    if cmd == "ort-eval":
        from moonshine_it import ort_eval
        return ort_eval.main(rest)
    if cmd == "validate":
        from moonshine_it import release
        return release.main(rest)
    if cmd == "parity":
        from moonshine_it import parity
        return parity.main(rest)
    print(f"unknown command '{cmd}'\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
