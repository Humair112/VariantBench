#!/usr/bin/env python3
import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_confusion_csv(path: str) -> pd.DataFrame:
    """
    Read a confusion matrix from a CSV with this shape:
        ,Pred1,Pred2,...
        True1,a,b,...
        True2,c,d,...
    The first column is the index (true labels). The header row are predicted labels.
    """
    df = pd.read_csv(path, index_col=0)
    # Coerce everything to numeric (non-numeric -> 0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def aggregate(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate multiple confusion matrices by computing the mean (and SD)."""
    dfs = [read_confusion_csv(p) for p in paths]
    if len(dfs) == 1:
        mean_df = dfs[0].copy()
        sd_df = pd.DataFrame(np.zeros_like(mean_df), index=mean_df.index, columns=mean_df.columns)
        return mean_df, sd_df

    all_rows = sorted(set().union(*[d.index for d in dfs]))
    all_cols = sorted(set().union(*[d.columns for d in dfs]))
    aligned = [d.reindex(index=all_rows, columns=all_cols, fill_value=0) for d in dfs]
    stack = np.stack([d.values for d in aligned], axis=0)  # (n, r, c)
    mean = pd.DataFrame(stack.mean(axis=0), index=all_rows, columns=all_cols)
    sd = pd.DataFrame(stack.std(axis=0, ddof=1), index=all_rows, columns=all_cols)
    return mean, sd


def normalize(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, bool]:
    """Normalize counts to percentages. Returns (normalized_df, is_percent_bool)."""
    if mode == "none":
        return df.copy(), False

    out = df.copy().astype(float)
    if mode == "rows":
        denom = out.sum(axis=1).replace(0, np.nan)
        out = out.div(denom, axis=0) * 100
    elif mode == "cols":
        denom = out.sum(axis=0).replace(0, np.nan)
        out = out.div(denom, axis=1) * 100
    elif mode == "all":
        total = out.values.sum()
        out = (out / total) * 100 if total > 0 else out
    else:
        raise ValueError("normalize must be one of: rows, cols, all, none")
    return out, True


def plot_heatmap(df: pd.DataFrame, title: str, outpath: str, annotate: str, fmt: str, sd_df: pd.DataFrame | None):
    """Make a single heatmap figure (matplotlib only; no seaborn, no custom colors)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(df.values, aspect="auto")  # default colormap

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)" if annotate == "perc" else "Count")

    # Axes
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)

    # Cell annotations
    data = df.values
    vmax = np.nanmax(data) if data.size else 0
    thresh = vmax / 2.0 if np.isfinite(vmax) and vmax > 0 else 0

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            if np.isnan(val):
                text = ""
            else:
                if annotate == "perc":
                    text = f"{val:{fmt}}"
                elif annotate == "counts":
                    text = f"{int(round(val))}"
                else:
                    text = ""
                # Optional SD on second line if provided
                if sd_df is not None and annotate != "none":
                    sd_val = sd_df.iat[i, j]
                    if np.isfinite(sd_val) and sd_val > 0:
                        text += f"\n±{sd_val:{fmt}}"
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if (np.isfinite(val) and val > thresh) else "black",
                fontsize=10
            )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build a confusion matrix heatmap from one or more CSVs."
    )
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="Path(s) to confusion matrix CSV file(s). You can pass multiple to average."
    )
    parser.add_argument(
        "--glob", action="store_true",
        help="Treat --csv arguments as glob patterns (e.g., results/*/matrix.csv)."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output image path. Defaults to '<csv>_heatmap.png' or 'confusion_heatmap.png' when given multiple CSVs."
    )
    parser.add_argument(
        "--title", default="Confusion Matrix Heatmap",
        help="Title text for the figure."
    )
    parser.add_argument(
        "--normalize", choices=["rows", "cols", "all", "none"], default="rows",
        help="How to normalize before plotting. Default: rows (row = true label → row sums to 100)."
    )
    parser.add_argument(
        "--annot", choices=["auto", "perc", "counts", "none"], default="auto",
        help="Annotation style. 'auto' = percentages if normalized, else counts."
    )
    parser.add_argument(
        "--labels", default="Benign,Likely Benign,Likely Pathogenic,Pathogenic,VUS",
        help="Comma-separated label order to enforce (e.g., 'Benign,Likely Benign,Likely Pathogenic,Pathogenic,VUS')."
    )
    parser.add_argument(
        "--with-sd", dest="with_sd", action="store_true",
        help="If multiple CSVs are provided, add ± SD on a second line."
    )
    parser.add_argument(
        "--fmt", default=".1f",
        help="Number format for percentages / SD (default: .1f)."
    )

    args = parser.parse_args()

    # Expand any globs if requested
    paths: List[str] = []
    for pat in args.csv:
        if args.glob or any(ch in pat for ch in "*?[]"):
            paths.extend(glob.glob(pat))
        else:
            paths.append(pat)

    if not paths:
        raise SystemExit("No CSV files matched.")

    # Aggregate
    mean_df, sd_df = aggregate(paths)

    # Reorder labels if requested
    if args.labels:
        order = [s.strip() for s in args.labels.split(",")]
        mean_df = mean_df.reindex(index=order, columns=order)
        sd_df = sd_df.reindex(index=order, columns=order)

    # Normalize
    norm_df, is_percent = normalize(mean_df, args.normalize)

    # Annotation style
    annotate_style = args.annot
    if annotate_style == "auto":
        annotate_style = "perc" if is_percent else "counts"

    # Default output name
    if args.out:
        out_path = args.out
    else:
        if len(paths) == 1:
            base = os.path.splitext(paths[0])[0]
            out_path = f"{base}_heatmap.png"
        else:
            out_path = "confusion_heatmap.png"

    # Plot
    sd_arg = sd_df if (args.with_sd and len(paths) > 1) else None
    plot_heatmap(
        norm_df,
        title=args.title,
        outpath=out_path,
        annotate=annotate_style,
        fmt=args.fmt,
        sd_df=sd_arg,
    )
    print(f"Saved heatmap to: {out_path}")


if __name__ == "__main__":
    main()
