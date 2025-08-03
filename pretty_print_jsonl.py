import json

filename = "results/gemini-2.5-flash_parsed.jsonl"   # <-- Change this if needed

with open(filename, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"Error on line {i}: {e}")
            print(line)
            continue

        print(f"\n--- Entry {i} ---")
        print(json.dumps(obj, indent=2))


