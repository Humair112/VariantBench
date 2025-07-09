import gzip

# Finding correct columns
with gzip.open(r"C:\variantbench\data\dbNSFP5.2a_grch38.gz", 'rt', encoding='utf-8', errors='replace') as f:
    header = f.readline().strip().split('\t')
    for i, col in enumerate(header):
        print(f"{i}\t{col}")
