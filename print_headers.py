#Printing file headers to make sure arguments match

with open("C:/variantbench/data/variant_summary.txt") as f:
    header = f.readline().strip().split('\t')
    print(header)
