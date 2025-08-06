import pandas as pd
import json
import csv
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--llm_jsonl", default='results/GeminiResults/gemini-2.5-flash_parsed.jsonl',
                    help="Path to LLM predictions JSONL (default: Track A output)")
parser.add_argument("--gold_csv", default='results/FrozenBenchmark/variantbench_100_gold.csv',
                    help="Path to gold-standard CSV")
parser.add_argument("--cm_csv", default='results/variantbench_confusion_matrix.csv',
                    help="Where to save the confusion matrix CSV")
parser.add_argument("--f1_csv", default='results/variantbench_per_flag_f1.csv',
                    help="Where to save the per-flag F1 and label report CSV")
args = parser.parse_args()

LLM_JSONL = args.llm_jsonl
GOLD_CSV  = args.gold_csv
CM_CSV    = args.cm_csv
F1_CSV    = args.f1_csv

FIELDS = ["label", "PM2", "PP3", "PS1", "BS1", "BA1"]

def load_llm_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            data[rec["vid"]] = rec
    return data

def load_gold_csv(path):
    df = pd.read_csv(path)
    df = df.set_index("variant")
    return df

llm = load_llm_jsonl(LLM_JSONL)
gold = load_gold_csv(GOLD_CSV)

labels_gold = []
labels_llm = []
per_flag_gold = {flag: [] for flag in FIELDS[1:]}
per_flag_llm = {flag: [] for flag in FIELDS[1:]}

for variant, row in gold.iterrows():
    llm_rec = llm.get(variant)
    labels_gold.append(row["label"])
    llm_label = llm_rec.get("label") if llm_rec else "MISSING"
    labels_llm.append(llm_label)
    for flag in FIELDS[1:]:
        gold_flag = row[flag]
        llm_flag = llm_rec.get(flag) if llm_rec else "MISSING"
        def norm(v):
            if isinstance(v, str):
                v = v.strip().lower()
                if v in {"true", "1"}: return True
                if v in {"false", "0"}: return False
            return bool(v)
        per_flag_gold[flag].append(norm(gold_flag))
        per_flag_llm[flag].append(norm(llm_flag))

# Confusion matrix for labels
unique_labels = sorted(set(labels_gold) | set(labels_llm))
cm = confusion_matrix(labels_gold, labels_llm, labels=unique_labels)
cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
cm_df.to_csv(CM_CSV)
print(f"Confusion matrix written to {CM_CSV}")

print("\nLabel Classification Report:\n")
print(classification_report(labels_gold, labels_llm, labels=unique_labels))

# Per-flag F1 scores
f1_rows = []
for flag in FIELDS[1:]:
    y_true = per_flag_gold[flag]
    y_pred = per_flag_llm[flag]
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    f1_rows.append({"flag": flag, "f1_score": f1})

# Get classification report as DataFrame
report_dict = classification_report(labels_gold, labels_llm, labels=unique_labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "flag"})

# Write both to the same CSV (flag F1 then a blank line then classification report)
with open(F1_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["flag", "f1_score"])
    writer.writeheader()
    writer.writerows(f1_rows)
    
    # Blank line for separation
    f.write("\n")
    
    # Write label classification report
    report_df.to_csv(f, index=False)

print(f"Combined per-flag F1 scores and label classification report written to {F1_CSV}")

print("\nConfusion matrix (as table):")
print(cm_df)

