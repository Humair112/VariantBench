import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score

FLAGS = ["PM2", "PP3", "PS1", "BS1", "BA1"]

def load_gold(gold_path: Path) -> pd.DataFrame:
    if gold_path.suffix.lower() == ".csv":
        df = pd.read_csv(gold_path)
    else:
        rows = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame(rows)
    if "variant" in df.columns and "variant_id" not in df.columns:
        df = df.rename(columns={"variant":"variant_id"})
    df["variant_id"] = df["variant_id"].astype(str)
    return df

def load_preds(pred_path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if "variant_id" not in df.columns:
        raise ValueError("predictions_converted.jsonl is missing 'variant_id'")
    df["variant_id"] = df["variant_id"].astype(str)
    return df

def detect_model(path: Path) -> str:
    s = str(path).lower()
    if "openai" in s or "gpt" in s:
        return "OpenAI"
    if "claude" in s:
        return "Claude"
    if "gemini" in s:
        return "Gemini"
    return "Unknown"

def main():
    ap = argparse.ArgumentParser(description="Compute Track B per-criterion F1 (PM2, PP3, PS1, BS1, BA1).")
    ap.add_argument("--gold", required=True, type=Path,
                    help="Path to variantbench_100_gold.csv (or ground_truth.jsonl).")
    ap.add_argument("--preds", required=True, type=Path,
                    help="Path to TrackB predictions_converted.jsonl (for OpenAI/Claude/Gemini).")
    ap.add_argument("--out", default=Path("trackB_per_flag_f1.csv"), type=Path,
                    help="Output CSV filename (default: trackB_per_flag_f1.csv).")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    preds = load_preds(args.preds)
    model_name = detect_model(args.preds)

    common = set(gold["variant_id"]).intersection(set(preds["variant_id"]))
    if not common:
        raise SystemExit("No overlapping variant_id values. Make sure predictions use chrom-pos-ref-alt IDs.")

    gts = gold.set_index("variant_id").loc[sorted(common)]
    prs = preds.set_index("variant_id").loc[sorted(common)]

    rows = []
    for flag in FLAGS:
        gcol = flag if flag in gts.columns else flag.lower()
        pcol = flag.lower() if flag.lower() in prs.columns else flag
        if gcol not in gts.columns or pcol not in prs.columns:
            continue
        y_true = gts[gcol].astype(bool).values
        y_pred = prs[pcol].astype(bool).values
        rows.append({"model": model_name, "flag": flag, "F1": f1_score(y_true, y_pred)})

    out_df = pd.DataFrame(rows).sort_values("flag")
    print(out_df.to_string(index=False))
    out_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
