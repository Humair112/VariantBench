import pandas as pd

# Update this path to your results file if needed for debugging
#df = pd.read_csv(r"C:\variantbench\results\acmg_algorithmic.csv")
df = pd.read_csv(r"C:\variantbench\results\acmg_debug.csv")

# Converting AF to float just in case
df['AF'] = df['AF'].astype(float)

# Defining the AF cutoffs in pipeline
PM2_cutoff = 0.0001
BS1_cutoff_low = 0.0001
BS1_cutoff_high = 0.05
BA1_cutoff = 0.05

print("\nRows where PM2 is TRUE and (AF is None or AF < 1e-4) (should be TRUE):")
print(df[(df["PM2"] == True) & ((df["AF"].isnull()) | (df["AF"] < 1e-4))][["variant", "AF", "PM2"]])

print("\nRows where PM2 is FALSE but AF is None or AF < 1e-4 (should be empty):")
print(df[(df["PM2"] == False) & ((df["AF"].isnull()) | (df["AF"] < 1e-4))][["variant", "AF", "PM2"]])

print("\nRows where PM2 is TRUE but AF >= 1e-4 (should be empty):")
print(df[(df["PM2"] == True) & (df["AF"] >= 1e-4)][["variant", "AF", "PM2"]])

print("\nRows where BA1 is TRUE and AF >= 0.05 (should be TRUE):")
print(df[(df['AF'] >= BA1_cutoff) & (df['BA1'] == True)])

print("\nRows where BA1 is FALSE but AF >= 0.05 (should be empty):")
print(df[(df['AF'] >= BA1_cutoff) & (df['BA1'] == False)])

print("\nRows where BS1 is TRUE and 0.0001 <= AF < 0.05 (should be TRUE):")
print(df[(df['AF'] >= BS1_cutoff_low) & (df['AF'] < BS1_cutoff_high) & (df['BS1'] == True)])

print("\nRows where BS1 is FALSE but 0.0001 <= AF < 0.05 (should be empty):")
print(df[(df['AF'] >= BS1_cutoff_low) & (df['AF'] < BS1_cutoff_high) & (df['BS1'] == False)])

print("\nRows where PP3 is TRUE:")
print(df[df["PP3"] == True][["variant", "SIFT_pred", "Polyphen2_HDIV_pred", "MutationTaster_pred",
                            "MutationAssessor_pred", "PROVEAN_pred", "MetaSVM_pred",
                            "MetaLR_pred", "REVEL_score", "PP3"]])

print("\nRows where PP3 is FALSE:")
print(df[df["PP3"] == False][["variant", "SIFT_pred", "Polyphen2_HDIV_pred", "MutationTaster_pred",
                             "MutationAssessor_pred", "PROVEAN_pred", "MetaSVM_pred",
                             "MetaLR_pred", "REVEL_score", "PP3"]])



