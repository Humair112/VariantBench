#!/usr/bin/env python
"""
Step 3 – Build LLM-ready prompts from the *public* inputs file.

Input  : results/variantbench_100_inputs.csv
Output : results/variantbench_100_prompts.jsonl  (one line = one JSON record)
         results/variantbench_100_prompts_preview.txt (human-readable preview)

The prompt instructs the model to return ONE JSON object with:
  - label  (one of: Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign)
  - PM2, PP3, PS1, BS1, BA1 (booleans)

We DO NOT leak our gold labels/flags here.
"""

import pandas as pd
import json
import pathlib
import textwrap
import argparse

ROOT       = pathlib.Path(__file__).resolve().parents[1]
RESULTS    = ROOT / "results"
INPUTS     = RESULTS / "variantbench_100_inputs.csv"
OUT_JSONL  = RESULTS / "variantbench_100_prompts.jsonl"
OUT_PREVIEW= RESULTS / "variantbench_100_prompts_preview.txt"

# 1. Prompt template (only what the model should see)

PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert clinical variant curator applying the ACMG/AMP guidelines.
    I will give you a single missense SNV with limited information:

    • HGVS (protein-level): {hgvs}
    • Allele frequency (gnomAD popmax): {af_popmax}
    • In-silico summary: {insilico} 

    Tasks:
    1. Assign ONE final 5-tier ACMG label: "Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", or "Benign".
    2. Decide TRUE/FALSE for the following ACMG criteria, using reasonable cutoffs:
       - PM2  (absent/ultra-rare in population databases)
       - PP3  (multiple in-silico tools support a deleterious effect)
       - PS1  (same amino-acid change as a known Pathogenic variant, but different DNA change)
       - BS1  (allele frequency is too high for the disorder)
       - BA1  (stand-alone benign frequency; very common)
    3. Return ONLY a single JSON object with this exact schema and lowercase booleans:

    {{
      "label": "<one of: Pathogenic|Likely Pathogenic|VUS|Likely Benign|Benign>",
      "PM2": true/false,
      "PP3": true/false,
      "PS1": true/false,
      "BS1": true/false,
      "BA1": true/false,
      "rationale": "<one short paragraph explaining your choices>"
    }}

    Do not include any extra keys or any extra text before or after the JSON.
    """)

# 2. Helpers

def fmt_float(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}g}"

def build_insilico(row: pd.Series) -> str:
    """Concise in-silico summary from whatever columns you kept."""
    parts = []

    if "CADD_phred" in row:
        parts.append(f"CADD={fmt_float(row['CADD_phred'], 1)}")

    if "SIFT_pred" in row:
        val = row["SIFT_pred"]
        parts.append(f"SIFT={val if pd.notna(val) else 'NA'}")

    if "Polyphen2_HDIV_pred" in row:
        val = row["Polyphen2_HDIV_pred"]
        parts.append(f"PolyPhen={val if pd.notna(val) else 'NA'}")

    if "MetaLR_score" in row and "MetaLR_pred" in row:
        score = fmt_float(row["MetaLR_score"], 2) if pd.notna(row["MetaLR_score"]) else "NA"
        pred  = row["MetaLR_pred"] if pd.notna(row["MetaLR_pred"]) else "NA"
        parts.append(f"MetaLR={score}({pred})")

    # Column names have a hyphen -> must use brackets
    if "fathmm-XF_coding_score" in row and "fathmm-XF_coding_pred" in row:
        score = fmt_float(row["fathmm-XF_coding_score"], 2) if pd.notna(row["fathmm-XF_coding_score"]) else "NA"
        pred  = row["fathmm-XF_coding_pred"] if pd.notna(row["fathmm-XF_coding_pred"]) else "NA"
        parts.append(f"FATHMM={score}({pred})")

    if "AlphaMissense_score" in row and pd.notna(row["AlphaMissense_score"]):
        parts.append(f"AlphaMissense={fmt_float(row['AlphaMissense_score'], 3)}")

    return "; ".join(parts) if parts else "NA"

def build_prompt_row(row: pd.Series) -> dict:
    """Return a dict with fields + final prompt string for this variant."""
    hgvs      = row["aa_change"]               
    af_popmax = fmt_float(row["AF_popmax"], 3)  # AF_popmax column in my inputs
    insilico  = build_insilico(row)

    prompt    = PROMPT_TEMPLATE.format(
        hgvs=hgvs,
        af_popmax=af_popmax,
        insilico=insilico,
    )

    record = {
        "vid": row["variant"],          # unique ID to track
        "hgvs": hgvs,
        "af_popmax": af_popmax,
        "insilico_summary": insilico,
        "prompt": prompt
    }

    # (Optional) keep truth columns here for downstream scoring scripts.
    # Just comment out if you truly don't want them in the JSONL.
    for col in ["PM2", "PP3", "PS1", "BS1", "BA1", "label"]:
        if col in row:
            record[col] = row[col]

    return record

# 3. Main
def main():
    print(f"Loading {INPUTS} …")
    df = pd.read_csv(INPUTS)

    # Required for the prompt builder
    needed = ["variant", "aa_change", "AF_popmax"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in inputs file: {missing}")

    # Build records
    records = [build_prompt_row(row) for _, row in df.iterrows()]

    # Write JSONL
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write preview text
    with OUT_PREVIEW.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec["prompt"] + "\n\n" + ("-" * 60) + "\n\n")

    print(f"✅ wrote {OUT_JSONL} ({len(records)} prompts)")
    print(f"👀 preview at {OUT_PREVIEW}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add --inputs or --out flags here if you want flexibility
    _ = parser.parse_args()
    main()
