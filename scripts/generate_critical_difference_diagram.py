import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_statistical_data():
    json_path = Path("results/statistical_analysis/statistical_tests.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Statistical tests not found at {json_path}.")
    with open(json_path, "r") as f:
        data = json.load(f)
    average_ranks = data["average_ranks"]
    cd = data["nemenyi"]["critical_distance"]
    cliques = data["cliques"]
    friedman_p = data["friedman"]["p_value"]
    kendalls_w = data["effect_size"]["kendalls_w"]
    print(f"Loaded data for {len(average_ranks)} models")
    print(f"Critical distance: {cd:.4f}")
    print(f"Friedman p={friedman_p:.4f}, Kendall's W={kendalls_w:.3f}")
    return average_ranks, cd, cliques, friedman_p, kendalls_w


def draw_cd_diagram(average_ranks, cd, cliques, friedman_p, kendalls_w):
    sorted_models = sorted(average_ranks.items(), key=lambda x: x[1])
    names = [m[0] for m in sorted_models]
    ranks = [m[1] for m in sorted_models]
    rank_min = 1.0
    rank_max = float(len(names))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(rank_min - 0.5, rank_max + 0.5)
    ax.set_ylim(-0.65, 1.05)
    ax.axis("off")

    axis_y = 0.0
    ax.plot([rank_min, rank_max], [axis_y, axis_y],
            color="black", linewidth=1.5, zorder=3)

    for r in range(int(rank_min), int(rank_max) + 1):
        ax.plot([r, r], [axis_y - 0.03, axis_y + 0.03],
                color="black", linewidth=1.2, zorder=3)
        ax.text(r, axis_y - 0.08, str(r), ha="center", va="top", fontsize=10)

    ax.text((rank_min + rank_max) / 2, axis_y - 0.22,
            "Average Rank  (lower = better)",
            ha="center", va="top", fontsize=11)

    label_heights = [0.20, 0.38, 0.20, 0.38, 0.20, 0.38, 0.20]
    for i, (name, rank) in enumerate(zip(names, ranks)):
        ax.plot(rank, axis_y, "o", color="black", markersize=6, zorder=5)
        y_label = axis_y + label_heights[i % len(label_heights)]
        ax.plot([rank, rank], [axis_y + 0.03, y_label - 0.03],
                color="black", linewidth=0.8, linestyle="--", zorder=2)
        ax.text(rank, y_label + 0.02, name,
                ha="center", va="bottom", fontsize=10,
                fontweight="bold" if i == 0 else "normal")

    cd_y = 0.88
    cd_start = rank_min
    cd_end = rank_min + cd
    ax.annotate("", xy=(cd_end, cd_y), xytext=(cd_start, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black",
                                lw=1.5, mutation_scale=12))
    ax.text((cd_start + cd_end) / 2, cd_y + 0.06,
            f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.9))

    clique_y_base = -0.20
    clique_y_step = 0.13
    for clique_idx, clique in enumerate(cliques):
        clique_ranks = [average_ranks[m] for m in clique]
        r_min = min(clique_ranks)
        r_max = max(clique_ranks)
        y = clique_y_base - clique_idx * clique_y_step
        ax.plot([r_min, r_max], [y, y],
                color="black", linewidth=4.5,
                solid_capstyle="round", zorder=4)
        for r in [r_min, r_max]:
            ax.plot([r, r], [y - 0.03, y + 0.03],
                    color="black", linewidth=2.0, zorder=4)

    stat_text = (
        f"Friedman p = {friedman_p:.4f},  "
        f"Kendall's W = {kendalls_w:.3f} (moderate effect)"
    )
    ax.text(0.5, -0.04, stat_text,
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9,
            style="italic", color="#333333")

    fig.tight_layout()
    return fig


def save_figure(fig, filename_base):
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename_base}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved PNG: {png_path}")

    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"  Saved PDF: {pdf_path}")

    eps_path = output_dir / f"{filename_base}.eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"  Saved EPS: {eps_path}")

    return pdf_path


def copy_to_reports(pdf_path, filename_base):
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    dest = reports_dir / f"{filename_base}.pdf"
    shutil.copy2(pdf_path, dest)
    print(f"  Copied to reports: {dest}")


def main():
    print("=" * 60)
    print("Critical Difference Diagram Generator")
    print("=" * 60)

    average_ranks, cd, cliques, friedman_p, kendalls_w = load_statistical_data()

    print("\nDrawing CD diagram...")
    fig = draw_cd_diagram(average_ranks, cd, cliques, friedman_p, kendalls_w)

    print("\nSaving figures...")
    filename_base = "critical_difference_diagram"
    pdf_path = save_figure(fig, filename_base)
    copy_to_reports(pdf_path, filename_base)

    plt.close(fig)
    print("\nDone.")
    print(
        "\nSuggested caption:\n"
        "Critical difference diagram (Demsar, 2006) comparing seven sentence "
        "embedding models across nine text classification datasets. "
        "Average ranks are shown on the horizontal axis (lower = better). "
        "The CD bar indicates the critical distance at alpha=0.05 (Nemenyi test). "
        "Models connected by a thick horizontal bar are not significantly "
        "different from each other. "
        f"Friedman test: p={friedman_p:.4f}; "
        f"Kendall's W={kendalls_w:.3f} (moderate effect)."
    )


if __name__ == "__main__":
    main()
