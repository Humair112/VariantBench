import pandas as pd, pathlib
from lib.helpers import sample_missense, PM2, BS1, BA1, PP3, PS1
from lib.combiner  import combine         


DEBUG = True  #Note: Turn off when not debugging



ROOT = pathlib.Path(__file__).parent
OUT  = ROOT / "results" / "acmg_algorithmic.csv"
OUT.parent.mkdir(exist_ok=True)

print("⏳ Sampling 10000 missense variants from dbNSFP …")
variants = sample_missense(10000)

print("⏳ Scoring ACMG rules …")

rows = []
for _, r in variants.iterrows():
    label_dict = {
        "PM2": PM2(r["gnomAD4.1_joint_POPMAX_AF"]),
        "BS1": BS1(r["gnomAD4.1_joint_POPMAX_AF"]),
        "BA1": BA1(r["gnomAD4.1_joint_POPMAX_AF"]),
        "PP3": PP3(r),
        "PS1": PS1(r),
    } 
    label = combine(label_dict)

    

    rows.append({
        "variant": f"{r.chrom}-{r.pos}-{r.ref}-{r.alt}",
        "aa_change": r.aa_change,
        "AF_popmax": r["gnomAD4.1_joint_POPMAX_AF"],
        "PM2": label_dict["PM2"],
        "BS1": label_dict["BS1"],
        "BA1": label_dict["BA1"],
        "PP3": label_dict["PP3"],
        "PS1": label_dict["PS1"],
        "label": combine(label_dict)
    })

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"✅ wrote {OUT}")

# File created for debugging
if DEBUG:
    debug_df = variants.copy()
    debug_df["variant"] = debug_df.chrom.astype(str) + "-" + debug_df.pos.astype(str) + "-" + debug_df.ref + "-" + debug_df.alt
    debug_df["aa_change"] = debug_df.aa_change
    debug_df["AF"] = debug_df.AF
    debug_df["PM2"] = debug_df["gnomAD4.1_joint_POPMAX_AF"].apply(PM2)
    debug_df["BS1"] = debug_df["gnomAD4.1_joint_POPMAX_AF"].apply(BS1)
    debug_df["BA1"] = debug_df["gnomAD4.1_joint_POPMAX_AF"].apply(BA1)
    debug_df["PP3"] = debug_df.apply(PP3, axis=1)
    debug_df["PS1"] = debug_df.apply(PS1, axis=1) 
    # After computing flags
    debug_df["label"] = debug_df.apply(
    lambda row: combine({
        "PM2": row["PM2"],
        "BS1": row["BS1"],
        "BA1": row["BA1"],
        "PP3": row["PP3"],
        "PS1": row["PS1"],
    }), axis=1
)

    debug_cols = [
        "variant", "aa_change", "AF",
        "gnomAD4.1_joint_POPMAX_AF",  
        "CADD_phred",
        "SIFT_pred",
        "Polyphen2_HDIV_pred",
        "MetaLR_score", "MetaLR_pred",
        "fathmm-XF_coding_score", "fathmm-XF_coding_pred",
        "AlphaMissense_score",
        "MutationTaster_pred", "MutationAssessor_pred", "PROVEAN_pred", "MetaSVM_pred", "REVEL_score",
        "PM2", "BS1", "BA1", "PP3", "PS1", "label"
    ]

    debug_df[debug_cols].to_csv(ROOT / "results" / "acmg_debug.csv", index=False)
    print(f"🪪 wrote {ROOT / 'results' / 'acmg_debug.csv'} [for PP3 debugging]")