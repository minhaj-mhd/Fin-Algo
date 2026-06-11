import os
import json
import shutil
import re

LEDGER_PATH = r"data\gauntlet\ledger.jsonl"
REPORTS_DIR = r"finalgo-memory-layer\finalgo\08. Model Analysis\Gauntlet Reports"
STATS_FILE = r"finalgo-memory-layer\finalgo\02. Model Suite\Model Performance & Statistics.md"

def sync_reports():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if not os.path.exists(LEDGER_PATH):
        print("Ledger not found.")
        return

    latest_runs = {}
    
    with open(LEDGER_PATH, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get('event') == 'completed' or 'verdicts' in record:
                    model = record.get('model_name')
                    run_id = record.get('run_id')
                    if model and run_id:
                        latest_runs[model] = record
            except Exception as e:
                pass
                
    for model, record in latest_runs.items():
        run_id = record['run_id']
        source_report = os.path.join(r"data\gauntlet", run_id, "report.md")
        dest_report = os.path.join(REPORTS_DIR, f"{model}_report.md")
        
        if os.path.exists(source_report):
            shutil.copy2(source_report, dest_report)
            print(f"Copied report for {model} to {dest_report}")
        else:
            print(f"Source report not found: {source_report}")

    print("Finished syncing reports.")

if __name__ == "__main__":
    sync_reports()
