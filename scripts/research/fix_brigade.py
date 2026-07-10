import os
import sys
import json
import pandas as pd

def main():
    # 1. Update manifest.json
    with open("manifest.json", "r") as f:
        manifest = json.load(f)

    if "BRIGADE.NS" in manifest["tickers"]:
        manifest["tickers"].remove("BRIGADE.NS")
    if "BRIGADE.NS" in manifest["missing_data"]:
        del manifest["missing_data"]["BRIGADE.NS"]

    manifest["excluded"] = ["BRIGADE.NS (corporate-action mismatch)"]

    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    # 2. Update coverage_report.md
    with open("coverage_report.md", "r") as f:
        lines = f.readlines()
        
    with open("coverage_report.md", "w") as f:
        for line in lines:
            if "BRIGADE.NS:" in line:
                f.write("- BRIGADE.NS: Excluded due to corporate-action mismatch.\n")
            else:
                f.write(line)

    # 3. Remove BRIGADE.csv from backfill
    brigade_path = "data/raw_upstox_cache_15min_3y_backfill/BRIGADE.csv"
    if os.path.exists(brigade_path):
        os.remove(brigade_path)
        
    print("Updated manifest, coverage report, and removed BRIGADE from backfill.")

if __name__ == '__main__':
    main()
