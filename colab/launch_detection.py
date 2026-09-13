import subprocess, sys

with open("/content/detect_train.log", "w") as f:
    pass

proc = subprocess.Popen(
    [sys.executable, "/content/train_detection.py"],
    stdout=open("/content/detect_train.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"DET_LAUNCHED pid={proc.pid}")