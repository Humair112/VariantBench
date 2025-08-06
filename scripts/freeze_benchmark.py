import pandas as pd
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Read debug variants from where your run_pipeline.py wrote them
DEBUG_CSV = ROOT / "results" / "CompleteACMGVariantsDataset" / "acmg_debug.csv"
df = pd.read_csv(DEBUG_CSV)

# Write all frozen benchmark outputs here:
FROZEN_BENCHMARK_DIR = ROOT / "results" / "FrozenBenchmark"
FROZEN_BENCHMARK_DIR.mkdir(exist_ok=True)

# Step 1: Sample 20 per label
N_PER_LABEL = 20
samples = []

for label in ["Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", "Benign"]:
    group = df[df["label"] == label]
    if len(group) < N_PER_LABEL:
        print(f"Warning: Only {len(group)} found for {label}")
        chosen = group
    else:
        chosen = group.sample(N_PER_LABEL, random_state=42)
    samples.append(chosen)

gold = pd.concat(samples).reset_index(drop=True)
# Optional: Check flag balance (print(gold[["PM2", "BS1", "BA1", "PP3", "PS1"]].mean()))

# Save full gold set with all columns
GOLD_CSV = FROZEN_BENCHMARK_DIR / "variantbench_100_gold.csv"
gold.to_csv(GOLD_CSV, index=False)
print(f"✅ wrote {GOLD_CSV}")

# Step 2: Make LLM input file
inputs = gold[[
    "variant", "aa_change", "gnomAD4.1_joint_POPMAX_AF", "CADD_phred",
    "SIFT_pred", "Polyphen2_HDIV_pred", "MetaLR_score", "MetaLR_pred",
    "fathmm-XF_coding_score", "fathmm-XF_coding_pred", "AlphaMissense_score"
]].copy()

inputs = inputs.rename(columns={"gnomAD4.1_joint_POPMAX_AF": "AF_popmax"})

INPUTS_CSV = FROZEN_BENCHMARK_DIR / "variantbench_100_inputs.csv"
inputs.to_csv(INPUTS_CSV, index=False)
print(f"✅ wrote {INPUTS_CSV}")
