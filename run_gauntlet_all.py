import os
import subprocess
import sys

models = [
    "v12_lambdamart_1h",
    "v13_ndcg_raw_1h"
]

dataset = "1h_v3_3y"

print(f"Starting Gauntlet test on {len(models)} models...")

for i, model in enumerate(models):
    print(f"\n[{i+1}/{len(models)}] Running Gauntlet on model: {model}")
    try:
        # Run the gauntlet command
        cmd = [sys.executable, "-m", "scripts.gauntlet.cli", "run", "--model", model, "--dataset", dataset]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[SUCCESS] {model}")
            # print the last few lines to show the result
            lines = result.stdout.splitlines()
            for line in lines[-5:]:
                print(line)
        else:
            print(f"[ERROR] {model}")
            print(result.stdout)
            print(result.stderr)
            
    except Exception as e:
        print(f"[EXCEPTION] {model}: {str(e)}")

print("\nFinished all model tests.")
