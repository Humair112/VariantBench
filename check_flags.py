import pandas as pd
from lib.helpers import PM2, BS1, BA1, PP3, PS1

df = pd.read_csv(r"C:\variantbench\results\FrozenBenchmark\variantbench_100_gold.csv")

if "gnomAD4.1_joint_POPMAX_AF" in df.columns:
    df = df.rename(columns={"gnomAD4.1_joint_POPMAX_AF": "AF_popmax"})

af_col = "AF_popmax"
required_cols = ["variant", af_col, "PM2", "BS1", "BA1", "PP3", "PS1"]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in file: {missing}")

df['PM2_check'] = df[af_col].apply(PM2)
df['BS1_check'] = df[af_col].apply(BS1)
df['BA1_check'] = df[af_col].apply(BA1)
df['PP3_check'] = df.apply(PP3, axis=1)
df['PS1_check'] = df.apply(PS1, axis=1)

flags = ["PM2", "BS1", "BA1", "PP3", "PS1"]

any_errors = False
for flag in flags:
    mismatches = df[df[flag] != df[f"{flag}_check"]]
    if not mismatches.empty:
        any_errors = True
        print(f"\n⚠️ {flag} mismatches ({len(mismatches)} rows):")
        print(mismatches[["variant", af_col, flag, f"{flag}_check"]])
    else:
        print(f"✅ {flag}: All values match!")

if not any_errors:
    print("\n🎉 All flags are correct and match your helpers.py logic.")
else:
    print("\n⚠️ There are mismatches above. Review those rows and double-check logic or data.")

import sys
if any_errors:
    sys.exit(1)
