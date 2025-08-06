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
import sys, pathlib as _pl

from lib.helpers import PS1




# 1. Prompt template (only what the model should see) (NO PS1 EVIDENCE)

PROMPT_TEMPLATE_TRACK_A = textwrap.dedent("""\
You are an expert clinical variant curator applying the ACMG/AMP guidelines.
I will give you a single missense SNV with limited information:

• HGVS (protein-level): {hgvs}
• Allele frequency (gnomAD popmax): {af_popmax}
• In-silico summary: {insilico}

Scope: Evaluate ONLY these five ACMG criteria and ignore all others:
- PM2  (absent/ultra-rare in population databases)
- PP3  (multiple in-silico tools support a deleterious effect)
- PS1  (same amino-acid change as a known Pathogenic variant, but different DNA change)
- BS1  (allele frequency is too high for the disorder)
- BA1  (stand-alone benign frequency; very common)

Tasks:
1) Assign ONE final 5-tier ACMG label: "Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", or "Benign".
2) Decide TRUE/FALSE for PM2, PP3, PS1, BS1, BA1 using only the information above. Do not invent fields not shown.

Return ONLY a single JSON object with this exact schema and lowercase booleans:

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

# Includes PS1 evidence
PROMPT_TEMPLATE_TRACK_B = textwrap.dedent("""\
You are an expert clinical variant curator applying the ACMG/AMP guidelines.
I will give you a single missense SNV with limited information:

• HGVS (protein-level): {hgvs}
• Allele frequency (gnomAD popmax): {af_popmax}
• In-silico summary: {insilico}
• PS1 evidence (ClinVar {clinvar_release}): {ps1_yes_no}  # "yes" or "no"

Scope: Evaluate ONLY these five ACMG criteria and ignore all others:
- PM2  (absent/ultra-rare in population databases)
- PP3  (multiple in-silico tools support a deleterious effect)
- PS1  (same amino-acid change as a known Pathogenic variant, but different DNA change)
- BS1  (allele frequency is too high for the disorder)
- BA1  (stand-alone benign frequency; very common)

Tasks:
1) Assign ONE final 5-tier ACMG label: "Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", or "Benign".
2) Decide TRUE/FALSE for PM2, PP3, PS1, BS1, BA1 using only the information above.
   - Set PS1 = true if and only if the PS1 evidence line is "yes"; otherwise PS1 = false.

Return ONLY a single JSON object with this exact schema and lowercase booleans:

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

   
    if "fathmm-XF_coding_score" in row and "fathmm-XF_coding_pred" in row:
        score = fmt_float(row["fathmm-XF_coding_score"], 2) if pd.notna(row["fathmm-XF_coding_score"]) else "NA"
        pred  = row["fathmm-XF_coding_pred"] if pd.notna(row["fathmm-XF_coding_pred"]) else "NA"
        parts.append(f"FATHMM={score}({pred})")

    if "AlphaMissense_score" in row and pd.notna(row["AlphaMissense_score"]):
        parts.append(f"AlphaMissense={fmt_float(row['AlphaMissense_score'], 3)}")

    

    return "; ".join(parts) if parts else "NA"

def build_prompt_row(row: pd.Series, track: str, clinvar_release: str) -> dict:
    """Return a dict with fields + final prompt string for this variant."""
    hgvs      = row["aa_change"]
    af_popmax = fmt_float(row["AF_popmax"], 3)
    insilico  = build_insilico(row)

    if track == "B":
        ps1_yes_no = "yes" if PS1(row) else "no"
        prompt = PROMPT_TEMPLATE_TRACK_B.format(
            hgvs=hgvs,
            af_popmax=af_popmax,
            insilico=insilico,
            clinvar_release=clinvar_release,
            ps1_yes_no=ps1_yes_no
        )
    else:
        prompt = PROMPT_TEMPLATE_TRACK_A.format(
            hgvs=hgvs,
            af_popmax=af_popmax,
            insilico=insilico
        )

    return {
        "vid": row["variant"],
        "hgvs": hgvs,
        "af_popmax": af_popmax,
        "insilico_summary": insilico,
        "prompt": prompt
    }

# 3. Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["A", "B"], default="A",
                        help="A = knowledge-only PS1; B = rule-application PS1 with yes/no evidence")
    parser.add_argument("--clinvar-release", default="Mar-2025",
                        help="Shown in the PS1 evidence line for Track B")
    args = parser.parse_args()

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    PROMPTS = ROOT / "results" / "prompts"
    PROMPTS.mkdir(parents=True, exist_ok=True)
    INPUTS = ROOT / "results" / "FrozenBenchmark" / "variantbench_100_inputs.csv"  

    OUT_JSONL  = PROMPTS / f"variantbench_100_prompts_track{args.track}.jsonl"
    OUT_PREVIEW= PROMPTS / f"variantbench_100_prompts_track{args.track}_preview.txt"

    print(f"Loading {INPUTS} …")
    df = pd.read_csv(INPUTS)

    needed = ["variant", "aa_change", "AF_popmax"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in inputs file: {missing}")

    records = [build_prompt_row(row, args.track, args.clinvar_release) for _, row in df.iterrows()]

    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with OUT_PREVIEW.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec["prompt"] + "\n\n" + ("-" * 60) + "\n\n")

    print(f"✅ wrote {OUT_JSONL} ({len(records)} prompts) [Track {args.track}]")
    print(f"👀 preview at {OUT_PREVIEW}")

