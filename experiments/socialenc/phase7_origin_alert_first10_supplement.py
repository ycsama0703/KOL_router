"""Supplement to phase7: first10 origin alert at thresholds 0.50 and 0.60.

This keeps the original phase7 script unchanged and fills the missing main-band
semantic thresholds between 0.45/0.55/0.65.
"""
from __future__ import annotations

import pathlib

import phase7_origin_alert as p7

p7.OUT = pathlib.Path(__file__).with_name("phase7_origin_alert_first10_supplement_result.json")
p7.THRESHOLDS = [0.50, 0.60]
p7.ORIGIN_WINDOWS = [
    {"name": "first10", "max_rank": 10},
]


if __name__ == "__main__":
    p7.main()
