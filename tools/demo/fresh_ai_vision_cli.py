#!/usr/bin/env python3
"""CLI wrapper for bounded fresh Hermes GPT5.5 vision smoke runs."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from ghostcaddie.video.fresh_ai_vision import main

if __name__ == "__main__":
    raise SystemExit(main())
