import subprocess, sys
print("Python:", sys.version.split()[0])
out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", out.stdout.strip() or "none")
print("SMOKE_TEST_OK")
