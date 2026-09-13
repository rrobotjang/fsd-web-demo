import sys
from pathlib import Path

log = Path("/content/detect_train.log")
if log.exists():
    lines = log.read_text().splitlines()
    tail = lines[-25:]
    print("\n".join(tail))
    print("=== DETECTION COMPLETE ===" if "ALL_DONE" in lines else
          "=== DETECTION FAILED ===" if "DETECTION_FAILED" in lines or "Traceback" in lines else
          "=== STILL RUNNING ===")
else:
    print("=== NO LOG YET ===")