"""Script to fix unicode escape sequences in a JSON file.

NOTE: Use this script only if you forgot to set ensure_ascii=False in json.dump().

Created by @pytholic on 2026.02.24
"""

import json

# Load the file with the \u codes
with open("data/finetuning/instruction_dataset.json", encoding="utf-8") as f:
    content = json.load(f)

# Save it back with clean symbols
with open("data/finetuning/instruction_dataset_readable.json", "w", encoding="utf-8") as f:
    json.dump(content, f, ensure_ascii=False, indent=2)
