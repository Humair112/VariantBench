import pandas as pd
import json
import csv
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--llm_jsonl", required=True, help="Path to LLM parsed JSONL file")
parser.add_argument("--gold_csv", default="results/FrozenBenchmark/variantbench_100_gold.csv")
parser.add_argument("--out_csv", default="results/variantbench_comparison_report.csv")
args = parser.parse_args()

LLM_JSONL = args.llm_jsonl
GOLD_CSV = args.gold_csv
OUT_CSV = args.out_csv


# Fields 
FIELDS = ["label", "PM2", "PP3", "PS1", "BS1", "BA1"]

# Loading LLM output 
def load_llm_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            data[rec["vid"]] = rec
    return data

# Loading gold file
def load_gold_csv(path):
    df = pd.read_csv(path)
    df = df.set_index("variant")
    return df

llm = load_llm_jsonl(LLM_JSONL)
gold = load_gold_csv(GOLD_CSV)

# Comparing and Saving results
rows = []
labels_gold = []
labels_llm = []
flags_gold = {k: [] for k in FIELDS[1:]}
flags_llm = {k: [] for k in FIELDS[1:]}
faithful_count = 0
faithful_total = 0

for variant, row in gold.iterrows():
    rec = {"variant": variant}
    llm_rec = llm.get(variant)
    if not llm_rec:
        for field in FIELDS:
            rec[f"{field}_gold"] = row[field]
            rec[f"{field}_llm"] = "MISSING"
            rec[f"{field}_status"] = "Missing in LLM output"
        rows.append(rec)
        continue

    for field in FIELDS:
        gold_val = row[field]
        llm_val = llm_rec.get(field, "MISSING")

        def norm(v):
            if isinstance(v, str):
                v = v.strip().lower()
                if v in {"true", "1"}: return True
                if v in {"false", "0"}: return False
            return v

        gold_val_norm = norm(gold_val)
        llm_val_norm = norm(llm_val)
        rec[f"{field}_gold"] = gold_val
        rec[f"{field}_llm"] = llm_val

        if llm_val == "MISSING":
            rec[f"{field}_status"] = "Missing in LLM output"
        elif gold_val_norm == llm_val_norm:
            rec[f"{field}_status"] = "Correct"
        else:
            rec[f"{field}_status"] = "Mismatch"

    labels_gold.append(str(row["label"]))
    labels_llm.append(str(llm_rec.get("label", "MISSING")))
    for f in FIELDS[1:]:
        flags_gold[f].append(bool(norm(row[f])))
        flags_llm[f].append(bool(norm(llm_rec.get(f, False))))

    # Faithfulness (Does rationale cite numeric evidence?)
    rationale = str(llm_rec.get("rationale", ""))
    for flag in FIELDS[1:]:
        if norm(llm_rec.get(flag, False)):
            faithful_total += 1
            
            if flag == "PM2" and any(k in rationale.lower() for k in ["frequency", "gnomad", "rare", "absent"]):
                faithful_count += 1
            elif flag == "BS1" and any(k in rationale.lower() for k in ["high frequency", "common", "too high"]):
                faithful_count += 1
            elif flag == "BA1" and any(k in rationale.lower() for k in ["very common", "stand-alone", "benign frequency"]):
                faithful_count += 1
            elif flag == "PP3" and any(k in rationale.lower() for k in ["in-silico", "deleterious", "prediction", "tool", "cadd", "sift", "polyphen"]):
                faithful_count += 1
            elif flag == "PS1" and any(k in rationale.lower() for k in ["same amino acid", "different dna change", "known pathogenic"]):
                faithful_count += 1

    rows.append(rec)



# METRICS 
def micro_f1(y_true_dict, y_pred_dict):
    all_true = []
    all_pred = []
    for f in y_true_dict:
        all_true.extend(y_true_dict[f])
        all_pred.extend(y_pred_dict[f])
    return f1_score(all_true, all_pred, average='micro', zero_division=0)

def macro_f1(y_true_dict, y_pred_dict):
    scores = []
    for f in y_true_dict:
        score = f1_score(y_true_dict[f], y_pred_dict[f], average='binary', zero_division=0)
        scores.append(score)
    return sum(scores) / len(scores)

print("\n==== METRICS ====")
# Label accuracy and MCC
label_acc = accuracy_score(labels_gold, labels_llm)
label_mcc = matthews_corrcoef(labels_gold, labels_llm)
print(f"Label Exact Match Accuracy: {label_acc:.3f}")
print(f"Label MCC: {label_mcc:.3f}")

# Micro/macro F1 on flags
micro = micro_f1(flags_gold, flags_llm)
macro = macro_f1(flags_gold, flags_llm)
print(f"Flag Micro-F1: {micro:.3f}")
print(f"Flag Macro-F1: {macro:.3f}")

# Faithfulness
faith_perc = 100 * faithful_count / faithful_total if faithful_total else 0
print(f"Faithfulness (numeric cues in rationale when flag True): {faithful_count}/{faithful_total} ({faith_perc:.1f}%)")


# Write to CSV
fieldnames = ["variant"]
for field in FIELDS:
    fieldnames += [f"{field}_gold", f"{field}_llm", f"{field}_status"]

with open(OUT_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

   
with open(OUT_CSV, "a", newline='', encoding='utf-8') as f:
    f.write("\n==== METRICS ====\n")
    f.write(f"Label Exact Match Accuracy: {label_acc:.3f}\n")
    f.write(f"Label MCC: {label_mcc:.3f}\n")
    f.write(f"Flag Micro-F1: {micro:.3f}\n")
    f.write(f"Flag Macro-F1: {macro:.3f}\n")
    f.write(f"Faithfulness (numeric cues in rationale when flag True): {faithful_count}/{faithful_total} ({faith_perc:.1f}%)\n")


print(f"Wrote comparison CSV to {OUT_CSV}")
