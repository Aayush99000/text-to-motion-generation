"""
visualize_tokens.py — Visualize generated RVQ motion tokens as heatmaps
========================================================================

Since our T2M-GPT model outputs RVQ tokens (not raw joint positions),
this script shows the token "fingerprint" of generated motions — a heatmap
of token IDs across 6 RVQ layers over time.

For actual stick-figure animation you would need the VQ-VAE decoder
(which maps tokens → 3D joint positions), not provided by the competition.

Run:
    python3 visualize_tokens.py \
        --csv submission_gpt200_sent_ep075_t0.6.csv \
        --n   6 \
        --out token_heatmaps.png
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── Token columns in submission CSV ──────────────────────────────────────────
TOKEN_COLS = ["base_tokens", "residual_1", "residual_2",
              "residual_3",  "residual_4", "residual_5"]
LAYER_LABELS = ["Layer 0 (base)", "Layer 1", "Layer 2",
                "Layer 3",        "Layer 4", "Layer 5 (fine)"]


def parse_tokens(row: pd.Series) -> np.ndarray:
    """Parse token columns from a CSV row → [6, T] int array."""
    layers = []
    for col in TOKEN_COLS:
        val = str(row.get(col, ""))
        if not val or val == "nan":
            return None
        ids = list(map(int, val.split()))
        layers.append(ids)
    # ensure all layers same length
    min_len = min(len(l) for l in layers)
    return np.array([l[:min_len] for l in layers], dtype=np.int32)  # [6, T]


def plot_sample(ax_heatmap, ax_hist, tokens: np.ndarray, text: str, sample_id):
    """Draw one sample: heatmap of token IDs + histogram of base layer."""
    n_layers, T = tokens.shape

    # ── Heatmap ───────────────────────────────────────────────────────────
    im = ax_heatmap.imshow(
        tokens, aspect="auto", interpolation="nearest",
        cmap="tab20b", vmin=0, vmax=511, origin="upper"
    )
    ax_heatmap.set_yticks(range(n_layers))
    ax_heatmap.set_yticklabels(LAYER_LABELS, fontsize=7)
    ax_heatmap.set_xlabel("Frame", fontsize=8)
    ax_heatmap.set_title(
        f'ID: {sample_id}\n"{text[:60]}{"…" if len(text)>60 else ""}"',
        fontsize=8, pad=3
    )
    ax_heatmap.tick_params(axis="x", labelsize=7)
    plt.colorbar(im, ax=ax_heatmap, fraction=0.03, pad=0.02,
                 label="Token ID (0–511)")

    # ── Base-layer token distribution ─────────────────────────────────────
    ax_hist.hist(tokens[0], bins=64, range=(0, 512),
                 color="#1565C0", alpha=0.8, edgecolor="none")
    ax_hist.set_xlabel("Token ID", fontsize=7)
    ax_hist.set_ylabel("Count", fontsize=7)
    ax_hist.set_title("Base layer distribution", fontsize=7)
    ax_hist.tick_params(labelsize=7)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="submission_gpt200_sent_ep075_t0.6.csv")
    parser.add_argument("--n",   type=int, default=6,
                        help="Number of samples to visualize")
    parser.add_argument("--out", default="token_heatmaps.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Check if it's a submission (test) or training CSV
    has_text = "sentence" in df.columns
    has_tokens = all(c in df.columns for c in TOKEN_COLS)

    if not has_tokens:
        print("ERROR: CSV does not have token columns. "
              "Expected: base_tokens, residual_1…5")
        return

    # Pick n evenly spaced samples
    indices = np.linspace(0, len(df) - 1, args.n, dtype=int)
    samples = df.iloc[indices]

    # ── Figure layout: 2 subplots per sample (heatmap + histogram) ───────
    fig = plt.figure(figsize=(18, args.n * 2.8))
    fig.suptitle(
        f"Generated Motion Token Heatmaps\n{args.csv}",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs_outer = gridspec.GridSpec(args.n, 1, figure=fig, hspace=0.6)

    for plot_idx, (_, row) in enumerate(samples.iterrows()):
        tokens = parse_tokens(row)
        if tokens is None:
            continue

        text = str(row.get("sentence", row.get("id", "?")))
        sample_id = row.get("id", plot_idx)

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_outer[plot_idx],
            width_ratios=[3, 1], wspace=0.3
        )
        ax_heat = fig.add_subplot(gs_inner[0])
        ax_hist = fig.add_subplot(gs_inner[1])

        plot_sample(ax_heat, ax_hist, tokens, text, sample_id)

    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved → {args.out}")

    # ── Also print stats ──────────────────────────────────────────────────
    print("\n── Token Statistics ──────────────────────────────────────────")
    all_base = []
    for _, row in df.iterrows():
        tokens = parse_tokens(row)
        if tokens is not None:
            all_base.extend(tokens[0].tolist())
    all_base = np.array(all_base)
    print(f"  Total frames generated : {len(all_base):,}")
    print(f"  Unique base tokens used: {np.unique(all_base).shape[0]} / 512")
    print(f"  Mean token ID          : {all_base.mean():.1f}")
    print(f"  Most common tokens     : {pd.Series(all_base).value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    main()
