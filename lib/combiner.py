"""
Light-weight ACMG/AMP combiner for 5 rules:
  BA1 (stand-alone benign),
  BS1 (strong benign),
  PS1 (strong pathogenic),
  PM2 (moderate pathogenic),
  PP3 (supporting pathogenic).

Order of precedence:
  1) Benign evidence first (BA1, BS1) – frequency trumps.
  2) Then pathogenic patterns.  
  3) Otherwise VUS.

This mirrors a trimmed InterVar/ACMG table and prevents conflicts like
BS1=TRUE but calling Likely Pathogenic.
"""

from typing import Dict

def combine(flags: Dict[str, bool]) -> str:
    BA1 = flags.get("BA1", False)
    BS1 = flags.get("BS1", False)
    PS1 = flags.get("PS1", False)
    PM2 = flags.get("PM2", False)
    PP3 = flags.get("PP3", False)

    # Benign side 
    if BA1:
        return "Benign"  # Stand-alone benign

    if BS1 and not (PS1 or PM2 or PP3):
        return "Likely Benign"

    if BS1 and (PS1 or PM2 or PP3):
        # Conflicting: frequency says benign, others say pathogenic
        return "VUS"

    # Pathogenic side 
    strong   = PS1
    moderate = PM2
    support  = PP3

    # Pathogenic (require more evidence with limited rule set)
    if strong and moderate and support:
        return "Pathogenic"

    # Likely Pathogenic combinations we can reach with our limited rules
    if strong and moderate:
        return "Likely Pathogenic"
    if strong and support:
        return "Likely Pathogenic"
    if moderate and support:
        return "Likely Pathogenic"

    # Strong alone is not enough; moderate alone or supporting alone not enough
    return "VUS"
