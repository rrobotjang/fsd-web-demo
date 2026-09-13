import sys
from pathlib import Path

log = Path("/content/lane_retrain.log")
if log.exists():
    lines = log.read_text().splitlines()
    print("\n".join(lines[-22:]))
    print("=== LANE2 DONE ===" if "ALL_DONE" in lines else
          "=== LANE2 FAILED ===" if "TRAINING_FAILED" in lines or "Traceback" in lines else
          "=== LANE2 RUNNING ===")
else:
    print("=== NO LOG YET ===")