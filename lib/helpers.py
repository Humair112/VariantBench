import duckdb, pandas as pd, pathlib, random
import gzip, functools, pathlib, csv
import re


CLINVAR_PATH = pathlib.Path(__file__).parent.parent / "data" / "variant_summary.txt"

three2one = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
    'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
    'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Ter':'*','Sec':'U','Pyl':'O','Xaa':'X'
}

def convert_to_short(prot_change: str) -> str | None:
    """
    Convert 'p.Arg132His' → 'p.R132H'. Return None if not a
    standard missense HGVS string.
    """
    m = re.match(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', prot_change)
    if not m:
        return None
    ref3, pos, alt3 = m.groups()
    if ref3 in three2one and alt3 in three2one:
        return f"p.{three2one[ref3]}{pos}{three2one[alt3]}"
    return None



@functools.lru_cache(maxsize=1)
def pathogenic_aa_changes() -> set[str]:
    ok = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
    aa_set: set[str] = set()
    with open(CLINVAR_PATH, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            if row["ClinicalSignificance"] not in ok:
                continue
            name = row["Name"]
            m = re.search(r'\((p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*))\)', name)
            if m:
                prot_change = m.group(1) # full "p.Cys282Tyr" 
                ref3 = m.group(2)
                pos  = m.group(3)
                alt3 = m.group(4)
                if ref3 in three2one and (alt3 in three2one or alt3 == '*'):
                    short = f"p.{three2one[ref3]}{pos}{three2one.get(alt3, '*')}"
                    # Filter: keep only missense, i.e. single aa change (not fs, ins, del, etc)
                    if '*' not in short and 'fs' not in prot_change and 'del' not in prot_change and 'ins' not in prot_change:
                        aa_set.add(short)
    return aa_set



BASE   = pathlib.Path(__file__).resolve().parents[1] / "data"
DBNSFP = BASE / "dbNSFP5.2a_grch38.gz"   # 40 GB file

# 1. Randomly sample X missense SNVs from dbNSFP

def sample_missense(n_samples: int = 100) -> pd.DataFrame:
    """
    Returns a DataFrame with n_samples random missense variants:
    chrom, pos, ref, alt, aa_change, AF, deleterious
    """
    con = duckdb.connect()

    # We define 'missense' as rows that have a different aaref vs aaalt

    query = f"""
SELECT
    "#chr" AS chrom,
    "pos(1-based)" AS pos,
    ref,
    alt,
    COALESCE("HGVSp_VEP", "HGVSp_snpEff") AS aa_change,
    TRY_CAST("gnomAD4.1_joint_AF" AS DOUBLE) AS AF,
    TRY_CAST("gnomAD4.1_joint_POPMAX_AF" AS DOUBLE) AS "gnomAD4.1_joint_POPMAX_AF",
    TRY_CAST(CADD_phred AS DOUBLE) AS CADD_phred,
    SIFT_pred,
    Polyphen2_HDIV_pred,
    MetaLR_pred,
    TRY_CAST(MetaLR_score AS DOUBLE) AS MetaLR_score,
    TRY_CAST("fathmm-XF_coding_score" AS DOUBLE) AS "fathmm-XF_coding_score",
    "fathmm-XF_coding_pred",
    TRY_CAST(AlphaMissense_score AS DOUBLE) AS AlphaMissense_score,
    MutationTaster_pred,
    MutationAssessor_pred,
    PROVEAN_pred,
    MetaSVM_pred,
    TRY_CAST(REVEL_score AS DOUBLE) AS REVEL_score
FROM read_csv_auto(
    '{DBNSFP}',
    delim='\t',
    header=TRUE,
    compression='gzip',
    types={{'#chr': 'VARCHAR'}},
    nullstr='.'
)
WHERE aaref NOT IN ('', '.')
  AND aaalt NOT IN ('', '.')
  AND aaref <> aaalt
  AND length(ref) = 1
  AND length(alt) = 1
  AND TRY_CAST("gnomAD4.1_joint_AF" AS DOUBLE) IS NOT NULL
ORDER BY random()
LIMIT {n_samples}
"""


    df = con.execute(query).df()
    return df


# 2. ACMG rule functions
def PM2(af_popmax: float | None) -> bool:
    return af_popmax is None or af_popmax < 1e-4   # “absent / exceedingly rare”

def BS1(af_popmax: float | None) -> bool:
    return af_popmax is not None and (0.0001 <= af_popmax < 0.05)


def BA1(af_popmax: float | None) -> bool:
    return af_popmax is not None and af_popmax >= 0.05  # “common” (stand-alone benign)

def PP3(row) -> bool:
    """
    Returns True if at least 3 out of 7 tools call 'damaging/deleterious', or if REVEL_score > 0.5
    """
    calls = 0
    if row.get('SIFT_pred') in ('D',): calls += 1
    if row.get('Polyphen2_HDIV_pred') in ('D',): calls += 1
    if row.get('MutationTaster_pred') in ('A', 'D'): calls += 1  # A: disease_causing_automatic, D: disease_causing
    if row.get('MutationAssessor_pred') in ('H', 'M'): calls += 1  # H: high, M: medium
    if row.get('PROVEAN_pred') in ('D',): calls += 1
    if row.get('MetaSVM_pred') in ('D',): calls += 1
    if row.get('MetaLR_pred') in ('D',): calls += 1
    if row.get('REVEL_score') is not None:
        try:
            if float(row.get('REVEL_score')) > 0.5: calls += 1
        except Exception:
            pass
    return calls >= 3

def PS1(row) -> bool:
    """
    True if ANY one of the true protein changes (from VEP/snpEff)
    matches a ClinVar pathogenic/likely_pathogenic amino-acid change.
    """
    raw = row.get("aa_change") or ""
    for prot in raw.split(";"):
        short = convert_to_short(prot.strip())
        if short and short in pathogenic_aa_changes():
            return True
    return False


