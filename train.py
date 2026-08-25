#!/usr/bin/env python
"""Training entry point: python train.py --profile smoke --hardware rocm12g"""

from moonshine_it.train_loop import main

if __name__ == "__main__":
    raise SystemExit(main())
