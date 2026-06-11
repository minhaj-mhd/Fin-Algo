import os
import re

file_path = r"c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\02. Model Suite\Model Performance & Statistics.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# For each row matching a model, we want to replace `Run: ...` with a link to the report
# Regex to match the model name in the first column, and then find the Run ID
def replacer(match):
    model_name = match.group(1)
    # The whole line
    line = match.group(0)
    
    # We map the display name to the actual model name for the report
    actual_model = model_name
    if model_name == "v14_lambdamart_no_es": actual_model = "v14_lambdamart_no_es_1h"
    if model_name == "v10_native_1h": actual_model = "v10_native_1h"
    
    report_link = f"[Report](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/08.%20Model%20Analysis/Gauntlet%20Reports/{actual_model}_report.md)"
    
    # Replace Run: ... with Run: ... | Report
    if "Run:" in line and "[Report]" not in line:
        line = re.sub(r"(Run: [^)]+)", r"\1 | " + report_link, line)
    return line

# Match rows starting with | **`model_name`**
new_content = re.sub(r"^\|\s*\*\*`([^`]+)`\*\*(.*?)$", replacer, content, flags=re.MULTILINE)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)
    
print("Updated links.")
