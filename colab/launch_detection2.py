import subprocess, sys

with open("/content/detect_train.log", "a") as f:
    f.write("\n--- Retry with canonical layout ---\n")

proc = subprocess.Popen(
    [sys.executable, "/content/train_detection2.py"],
    stdout=open("/content/detect_train.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"DET2_LAUNCHED pid={proc.pid}")