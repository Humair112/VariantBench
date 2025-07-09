import pandas as pd, pathlib
from lib.helpers import sample_missense, PM2, BS1, BA1, PP3, PS1

DEBUG = True  #Note: Turn off when not debugging


ROOT = pathlib.Path(__file__).parent
OUT  = ROOT / "results" / "acmg_algorithmic.csv"
OUT.parent.mkdir(exist_ok=True)

print("⏳ Sampling 200 missense variants from dbNSFP …")
variants = sample_missense(200)

print("⏳ Scoring ACMG rules …")
rows = []
for _, r in variants.iterrows(): 
    rows.append({
        "variant": f"{r.chrom}-{r.pos}-{r.ref}-{r.alt}",
        "aa_change": r.aa_change,
        "AF": r.AF,
        "PM2": PM2(r.AF),
        "BS1": BS1(r.AF),
        "BA1": BA1(r.AF),
        "PP3": PP3(r),
        "PS1": PS1(r),     

    })

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"✅ wrote {OUT}")


# File created for debugging
if DEBUG:
    debug_df = variants.copy()
    debug_df["variant"] = debug_df.chrom.astype(str) + "-" + debug_df.pos.astype(str) + "-" + debug_df.ref + "-" + debug_df.alt
    debug_df["aa_change"] = debug_df.aa_change
    debug_df["AF"] = debug_df.AF
    debug_df["PM2"] = debug_df["AF"].apply(PM2)
    debug_df["BS1"] = debug_df["AF"].apply(BS1)
    debug_df["BA1"] = debug_df["AF"].apply(BA1)
    debug_df["PP3"] = debug_df.apply(PP3, axis=1)
    debug_df["PS1"] = debug_df.apply(PS1, axis=1)
    debug_cols = [
        "variant", "aa_change", "AF",
        "SIFT_pred", "Polyphen2_HDIV_pred", "MutationTaster_pred",
        "MutationAssessor_pred", "PROVEAN_pred", "MetaSVM_pred",
        "MetaLR_pred", "REVEL_score", "PM2", "BS1", "BA1", "PP3", "PS1"
    ]
    debug_df[debug_cols].to_csv(ROOT / "results" / "acmg_debug.csv", index=False)
    print(f"🪪 wrote {ROOT / 'results' / 'acmg_debug.csv'} [for PP3 debugging]")