"""
Parallel Batch Validation Gauntlet Runner (v10 to v19)
======================================================
Runs the Validation Gauntlet with --step-months 2 in parallel.
Uses ThreadPoolExecutor to run up to 3 models concurrently.
"""

import os
import sys
import subprocess
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS = [
    "v10_native_1h",
    "v10_depth4_1h",
    "v11_utility_1h",
    "v12_lambdamart_1h",
    "v13_ndcg_raw_1h",
    "v14_lambdamart_no_es_1h",
    "v15_lambdamart_es_1h",
    "v15_lambdamart_map5_1h",
    "v16_binary_breakout_1h",
    "v17_random_forest_1h",
    "v18_random_forest_1h",
    "v19_catboost_1h",
]

CONCURRENCY = 3

def run_model_gauntlet(model_name: str):
    print(f"[{model_name}] Starting Gauntlet run...")
    
    python_exe = os.path.join("env", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    cmd = [
        python_exe,
        "-m",
        "scripts.gauntlet.cli",
        "run",
        "--model",
        model_name,
        "--dataset",
        "1h_v3_3y",
        "--step-months",
        "2"
    ]
    
    start_time = time.time()
    try:
        # 12 minutes timeout per model
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=720
        )
        elapsed = time.time() - start_time
        
        stdout = process.stdout
        stderr = process.stderr
        
        # Log outputs
        log_dir = os.path.join("data", "gauntlet_batch_logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{model_name}_stdout.log"), "w", encoding="utf-8") as f:
            f.write(stdout)
        if stderr:
            with open(os.path.join(log_dir, f"{model_name}_stderr.log"), "w", encoding="utf-8") as f:
                f.write(stderr)
                
        if process.returncode != 0:
            print(f"[{model_name}] FAIL (Exited with code {process.returncode} in {elapsed:.1f}s)")
            err_msg = "Unknown error"
            for line in (stderr.splitlines() + stdout.splitlines()):
                if "Error" in line or "Exception" in line or "RuntimeError" in line or "ValueError" in line:
                    err_msg = line.strip()
            return {
                "model": model_name,
                "status": "FAIL",
                "returncode": process.returncode,
                "error": err_msg,
                "elapsed": elapsed,
                "run_id": "N/A",
                "verdicts": {}
            }
            
        # Parse run_id and verdicts from stdout
        run_id = "N/A"
        verdicts = {}
        for line in stdout.splitlines():
            if "[SUCCESS] Run completed:" in line:
                run_id = line.split(":")[-1].strip()
            if "Verdicts:" in line:
                try:
                    verdicts_str = line.split(":", 1)[1].strip()
                    verdicts = json.loads(verdicts_str.replace("'", '"'))
                except Exception:
                    pass
                    
        print(f"[{model_name}] SUCCESS (Completed in {elapsed:.1f}s | Run ID: {run_id} | Verdicts: {verdicts})")
        return {
            "model": model_name,
            "status": "SUCCESS",
            "returncode": 0,
            "elapsed": elapsed,
            "run_id": run_id,
            "verdicts": verdicts
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[{model_name}] TIMEOUT (Terminated after {elapsed:.1f}s)")
        return {
            "model": model_name,
            "status": "TIMEOUT",
            "returncode": -1,
            "error": "Execution timed out (12 minutes)",
            "elapsed": elapsed,
            "run_id": "N/A",
            "verdicts": {}
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{model_name}] ERROR (Exception in {elapsed:.1f}s: {e})")
        return {
            "model": model_name,
            "status": "ERROR",
            "returncode": -2,
            "error": str(e),
            "elapsed": elapsed,
            "run_id": "N/A",
            "verdicts": {}
        }

def main():
    print("=" * 80)
    print(f"PARALLEL BATCH RUNNER STARTING (CONCURRENCY={CONCURRENCY})")
    print("=" * 80)
    
    start_total = time.time()
    results = {}
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(run_model_gauntlet, m): m for m in MODELS}
        
        for future in as_completed(futures):
            model = futures[future]
            try:
                res = future.result()
                results[model] = res
            except Exception as e:
                print(f"[{model}] Future raised exception: {e}")
                results[model] = {
                    "model": model,
                    "status": "FUTURE_ERROR",
                    "returncode": -3,
                    "error": str(e),
                    "elapsed": 0.0,
                    "run_id": "N/A",
                    "verdicts": {}
                }
                
    elapsed_total = time.time() - start_total
    
    # Write summary report
    print("\n" + "=" * 80)
    print(f"BATCH RUN COMPLETED IN {elapsed_total:.1f}s")
    print("=" * 80)
    
    report_lines = [
        "# Gauntlet Parallel Batch Run (v10 to v19 with step_months=2)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Elapsed Time**: {elapsed_total:.1f} seconds",
        "",
        "| Model Name | Status | Run ID | Long Verdict | Short Verdict | Time (s) | Notes/Error |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]
    
    for model in MODELS:
        res = results.get(model, {"status": "MISSING", "run_id": "N/A", "elapsed": 0.0})
        status = res["status"]
        run_id = res["run_id"]
        elapsed = f"{res['elapsed']:.1f}"
        verdicts = res.get("verdicts", {})
        long_v = verdicts.get("long", "N/A")
        short_v = verdicts.get("short", "N/A")
        notes = res.get("error", "")
        
        report_lines.append(
            f"| `{model}` | {status} | `{run_id}` | {long_v} | {short_v} | {elapsed} | {notes} |"
        )
        
    report_content = "\n".join(report_lines)
    print(report_content)
    
    report_path = os.path.join("data", "gauntlet_batch_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nSaved batch run report to {report_path}")

if __name__ == "__main__":
    main()
