"""
visualization.py — Training loss curves for all KSL text-to-motion models
==========================================================================

Run:
    python visualization.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

# T2M-GPT sentence-only (100 epochs)
gpt_sentence = [
    3.8284, 2.4645, 2.0417, 1.7974, 1.6299, 1.5047, 1.4065, 1.3285, 1.2632, 1.2086,
    1.1601, 1.1193, 1.0815, 1.0491, 1.0172, 0.9894, 0.9643, 0.9163, 0.9376, 0.9164,
    0.8959, 0.8771, 0.8594, 0.8425, 0.8269, 0.8119, 0.7979, 0.7834, 0.7707, 0.7589,
    0.7480, 0.7353, 0.7248, 0.7149, 0.6822, 0.7150, 0.7077, 0.6993, 0.6917, 0.6847,
    0.6773, 0.6696, 0.6633, 0.6563, 0.6485, 0.6420, 0.6350, 0.6298, 0.6224, 0.6160,
    0.6103, 0.5826, 0.6182, 0.6151, 0.6105, 0.6065, 0.6026, 0.5978, 0.5936, 0.5896,
    0.5856, 0.5804, 0.5757, 0.5709, 0.5672, 0.5622, 0.5571, 0.5536, 0.5279, 0.5641,
    0.5635, 0.5602, 0.5572, 0.5550, 0.5513, 0.5487, 0.5461, 0.5423, 0.5384, 0.5347,
    0.5311, 0.5283, 0.5243, 0.5198, 0.5170, 0.4929, 0.5299, 0.5298, 0.5274, 0.5256,
    0.5234, 0.5211, 0.5181, 0.5163, 0.5135, 0.5097, 0.5069, 0.5039, 0.5013, 0.4981,
]

# T2M-GPT gloss+sentence (200 epochs)
gpt_both = [
    3.8279, 2.4632, 2.0305, 1.7804, 1.6091, 1.4828, 1.3840, 1.3052, 1.2400, 1.1852,
    1.1380, 1.0969, 1.0613, 1.0291, 0.9989, 0.9723, 0.9483, 0.8898, 0.9075, 0.8865,
    0.8673, 0.8494, 0.8330, 0.8168, 0.8024, 0.7881, 0.7755, 0.7630, 0.7516, 0.7412,
    0.7314, 0.7213, 0.7123, 0.6691, 0.6957, 0.6876, 0.6802, 0.6730, 0.6662, 0.6587,
    0.6519, 0.6456, 0.6399, 0.6342, 0.6288, 0.6235, 0.6185, 0.6137, 0.6089, 0.5721,
    0.6020, 0.5981, 0.5942, 0.5903, 0.5869, 0.5831, 0.5784, 0.5751, 0.5719, 0.5680,
    0.5650, 0.5621, 0.5590, 0.5556, 0.5530, 0.5191, 0.5499, 0.5473, 0.5449, 0.5429,
    0.5406, 0.5377, 0.5351, 0.5329, 0.5304, 0.5282, 0.5258, 0.5236, 0.5221, 0.5194,
    0.5171, 0.4855, 0.5157, 0.5143, 0.5125, 0.5114, 0.5099, 0.5076, 0.5060, 0.5041,
    0.5022, 0.5002, 0.4988, 0.4968, 0.4959, 0.4939, 0.4918, 0.5438, 0.4647, 0.4931,
    0.4915, 0.4905, 0.4892, 0.4881, 0.4862, 0.4848, 0.4830, 0.4818, 0.4802, 0.4788,
    0.4770, 0.4761, 0.4748, 0.4728, 0.5007, 0.4433, 0.4732, 0.4723, 0.4718, 0.4708,
    0.4696, 0.4690, 0.4678, 0.4661, 0.4651, 0.4635, 0.4623, 0.4612, 0.4604, 0.4588,
    0.4573, 0.4299, 0.4574, 0.4561, 0.4556, 0.4545, 0.4540, 0.4535, 0.4524, 0.4505,
    0.4504, 0.4491, 0.4480, 0.4466, 0.4461, 0.4454, 0.4437, 0.4830, 0.4177, 0.4460,
    0.4455, 0.4453, 0.4443, 0.4437, 0.4428, 0.4422, 0.4407, 0.4401, 0.4388, 0.4382,
    0.4367, 0.4363, 0.4355, 0.4335, 0.4561, 0.4061, 0.4357, 0.4356, 0.4349, 0.4343,
    0.4336, 0.4334, 0.4330, 0.4315, 0.4308, 0.4294, 0.4294, 0.4280, 0.4274, 0.4265,
    0.4245, 0.4407, 0.3969, 0.4265, 0.4269, 0.4260, 0.4263, 0.4253, 0.4248, 0.4244,
    0.4236, 0.4228, 0.4214, 0.4214, 0.4199, 0.4198, 0.4185, 0.4168, 0.3923, 0.4182,
]

# MoMask baselines (approximate from logs)
momask_sentence = {1: 4.2, 10: 2.8, 20: 2.2, 30: 1.9, 40: 1.75, 50: 1.65,
                   60: 1.58, 70: 1.52, 80: 1.48, 90: 1.45, 100: 1.43}
momask_gloss    = {1: 4.0, 10: 2.6, 20: 2.0, 30: 1.72, 40: 1.57, 50: 1.47,
                   60: 1.41, 70: 1.37, 80: 1.35, 90: 1.34, 100: 1.33}

# ── Smoothing helper ──────────────────────────────────────────────────────────

def smooth(values, window=5):
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("KSL Text-to-Motion — Training Loss Curves", fontsize=15, fontweight="bold", y=1.01)

COLORS = {
    "momask_s"  : "#9E9E9E",
    "momask_g"  : "#BDBDBD",
    "gpt_s"     : "#1565C0",
    "gpt_both"  : "#C62828",
}

# ── Left: all models overlaid ─────────────────────────────────────────────────
ax = axes[0]
ax.set_title("All Models — Training Loss", fontsize=12, fontweight="bold")

# MoMask (sparse points → interpolate)
mm_s_x = list(momask_sentence.keys())
mm_s_y = list(momask_sentence.values())
mm_g_x = list(momask_gloss.keys())
mm_g_y = list(momask_gloss.values())
ax.plot(mm_s_x, mm_s_y, "o--", color=COLORS["momask_s"], linewidth=1.5,
        markersize=4, label="MoMask (sentence)", alpha=0.8)
ax.plot(mm_g_x, mm_g_y, "s--", color=COLORS["momask_g"], linewidth=1.5,
        markersize=4, label="MoMask (gloss)", alpha=0.8)

# GPT sentence-only
x_s = list(range(1, len(gpt_sentence) + 1))
ax.plot(x_s, gpt_sentence, color=COLORS["gpt_s"], linewidth=1, alpha=0.3)
ax.plot(x_s, smooth(gpt_sentence), color=COLORS["gpt_s"], linewidth=2.5,
        label="T2M-GPT (sentence, 100ep)")

# GPT gloss+sentence
x_b = list(range(1, len(gpt_both) + 1))
ax.plot(x_b, gpt_both, color=COLORS["gpt_both"], linewidth=1, alpha=0.3)
ax.plot(x_b, smooth(gpt_both, window=7), color=COLORS["gpt_both"], linewidth=2.5,
        label="T2M-GPT (gloss+sent, 200ep)")

# Final loss annotations
ax.annotate(f"1.43", xy=(100, 1.43), xytext=(85, 1.55),
            fontsize=8, color=COLORS["momask_s"],
            arrowprops=dict(arrowstyle="->", color=COLORS["momask_s"], lw=1))
ax.annotate(f"1.33", xy=(100, 1.33), xytext=(85, 1.20),
            fontsize=8, color=COLORS["momask_g"],
            arrowprops=dict(arrowstyle="->", color=COLORS["momask_g"], lw=1))
ax.annotate(f"0.498", xy=(100, 0.498), xytext=(75, 0.38),
            fontsize=8, color=COLORS["gpt_s"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gpt_s"], lw=1))
ax.annotate(f"0.392\n(ep199)", xy=(199, 0.3923), xytext=(160, 0.28),
            fontsize=8, color=COLORS["gpt_both"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gpt_both"], lw=1))

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Training Loss", fontsize=11)
ax.set_xlim(0, 205)
ax.set_ylim(0.2, 4.5)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Right: GPT models zoomed in ───────────────────────────────────────────────
ax2 = axes[1]
ax2.set_title("T2M-GPT Models — Zoomed In", fontsize=12, fontweight="bold")

ax2.plot(x_s, gpt_sentence, color=COLORS["gpt_s"], linewidth=1, alpha=0.25)
ax2.plot(x_s, smooth(gpt_sentence), color=COLORS["gpt_s"], linewidth=2.5,
         label="T2M-GPT sentence (100ep) — final: 0.498")

ax2.plot(x_b, gpt_both, color=COLORS["gpt_both"], linewidth=1, alpha=0.25)
ax2.plot(x_b, smooth(gpt_both, window=7), color=COLORS["gpt_both"], linewidth=2.5,
         label="T2M-GPT gloss+sent (200ep) — best: 0.392")

# Shade SLURM restart spikes
restart_epochs = [18, 35, 52, 66, 82, 99, 116, 132, 149, 166, 183, 199]
for ep in restart_epochs:
    ax2.axvline(x=ep, color="orange", linewidth=0.7, linestyle=":", alpha=0.6)
restart_patch = mpatches.Patch(color="orange", alpha=0.4, label="SLURM job restart (LR warmup reset)")
ax2.axvline(x=restart_epochs[0], color="orange", linewidth=0.7, linestyle=":", alpha=0.6)

# Kaggle score annotations
ax2.text(102, 0.51, "Kaggle: 0.433", fontsize=8, color=COLORS["gpt_s"],
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["gpt_s"], alpha=0.8))
ax2.text(160, 0.40, "Kaggle: 0.425\n(ep200)", fontsize=8, color=COLORS["gpt_both"],
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["gpt_both"], alpha=0.8))

handles, labels = ax2.get_legend_handles_labels()
handles.append(restart_patch)
labels.append("SLURM restart (LR warmup reset)")
ax2.legend(handles=handles, labels=labels, fontsize=8.5, loc="upper right")

ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Training Loss", fontsize=11)
ax2.set_xlim(0, 205)
ax2.set_ylim(0.3, 1.1)
ax2.grid(True, alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Saved → training_curves.png")

# ── Also save a compact summary plot ─────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(10, 5))
ax3.set_title("Training Loss — All Models Comparison", fontsize=13, fontweight="bold")

ax3.plot(mm_s_x, mm_s_y, "o--", color=COLORS["momask_s"], linewidth=2,
         markersize=5, label="MoMask sentence — final: 1.43  |  Kaggle: 0.405")
ax3.plot(mm_g_x, mm_g_y, "s--", color=COLORS["momask_g"], linewidth=2,
         markersize=5, label="MoMask gloss    — final: 1.33  |  Kaggle: 0.404")
ax3.plot(x_s, smooth(gpt_sentence), color=COLORS["gpt_s"], linewidth=2.5,
         label="T2M-GPT sentence — final: 0.498  |  Kaggle: 0.433 ★")
ax3.plot(x_b, smooth(gpt_both, window=7), color=COLORS["gpt_both"], linewidth=2.5,
         label="T2M-GPT gloss+sent — best: 0.392  |  Kaggle: 0.425")

ax3.axhline(y=0.498, color=COLORS["gpt_s"],  linewidth=0.8, linestyle="--", alpha=0.4)
ax3.axhline(y=0.392, color=COLORS["gpt_both"], linewidth=0.8, linestyle="--", alpha=0.4)

ax3.set_xlabel("Epoch", fontsize=11)
ax3.set_ylabel("Training Loss", fontsize=11)
ax3.set_xlim(0, 205)
ax3.set_ylim(0.2, 4.6)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("training_summary.png", dpi=150, bbox_inches="tight")
print("Saved → training_summary.png")
