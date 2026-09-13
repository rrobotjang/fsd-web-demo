from pathlib import Path

log = Path("/content/train.log")
if not log.exists():
    print("NO_LOG")
else:
    lines = log.read_text(errors="replace").splitlines()
    print("\n".join(lines[-25:]))
    if any("ALL_DONE" in l for l in lines):
        print("=== TRAINING COMPLETE ===")
    if any("TRAINING_FAILED" in l or "Traceback" in l for l in lines):
        print("=== TRAINING FAILED ===")