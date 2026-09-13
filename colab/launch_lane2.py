import subprocess, sys

with open("/content/lane_retrain.log", "w") as f:
    pass

proc = subprocess.Popen(
    [sys.executable, "/content/retrain_lane.py"],
    stdout=open("/content/lane_retrain.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"LANE2_LAUNCHED pid={proc.pid}")