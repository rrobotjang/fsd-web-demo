import subprocess, sys

# Launch training in background so exec returns immediately
with open("/content/train.log", "w") as f:
    pass

proc = subprocess.Popen(
    [sys.executable, "/content/train_lane.py"],
    stdout=open("/content/train.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"LAUNCHED pid={proc.pid}")
print("Poll /content/train.log for progress")