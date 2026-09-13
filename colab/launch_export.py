import subprocess, sys

with open("/content/train.log", "a") as f:
    f.write("\n--- Export phase ---\n")

proc = subprocess.Popen(
    [sys.executable, "/content/export_lane.py"],
    stdout=open("/content/train.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"EXPORT_LAUNCHED pid={proc.pid}")