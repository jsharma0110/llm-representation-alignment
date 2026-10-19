import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path(
    "results/llama1b_llama3b/profiling"
)

FIGURES_DIR = Path(
    "figures/llama1b_llama3b"
)

MODEL_FILES = {
    "Llama-3.2-1B": RESULTS_DIR / "profile_llama1b.json",
    "Llama-3.2-3B": RESULTS_DIR / "profile_llama3b.json",
}


def load_results():
    """Load profiling results produced by run_profile.py."""

    results = {}

    for label, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find profiling result: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            results[label] = json.load(f)

    return results


def plot_forward_latency(results):
    """
    Plot mean steady-state forward latency for Llama-3.2-1B
    and Llama-3.2-3B.

    Mean latency is calculated over 100 measured forward passes
    after 10 warmup passes.
    """

    labels = list(results.keys())

    mean_latency = [
        results[label]["mean_forward_ms"]
        for label in labels
    ]

    # Calculate scale-up cost dynamically from results.
    small_latency = mean_latency[0]
    large_latency = mean_latency[1]

    latency_ratio = large_latency / small_latency

    latency_increase_percent = (
        (large_latency - small_latency)
        / small_latency
        * 100
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.5, 5.4)
    )

    bars = ax.bar(
        labels,
        mean_latency,
        width=0.55,
    )

    # ---------------------------------------------------------
    # Titles
    # ---------------------------------------------------------

    fig.suptitle(
        "Inference Cost Increases with Model Scale",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    ax.set_title(
        "Llama-3.2 forward latency on NVIDIA H200 NVL",
        fontsize=11,
        pad=12,
    )

    # ---------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------

    ax.set_ylabel(
        "Mean forward latency (ms)",
        fontsize=12,
    )

    ax.tick_params(
        axis="x",
        labelsize=11,
    )

    ax.tick_params(
        axis="y",
        labelsize=10,
    )

    # Start at zero because this is a magnitude comparison.
    ax.set_ylim(
        0,
        max(mean_latency) * 1.35,
    )

    # Remove unnecessary borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Subtle horizontal guide lines.
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    # ---------------------------------------------------------
    # Values above bars
    # ---------------------------------------------------------

    for bar, value in zip(
        bars,
        mean_latency,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.10,
            f"{value:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # ---------------------------------------------------------
    # Main takeaway annotation
    # ---------------------------------------------------------

    takeaway = (
        f"3B: {latency_ratio:.2f}× higher "
        f"mean latency  (+{latency_increase_percent:.1f}%)"
    )

    ax.text(
        0.5,
        0.88,
        takeaway,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.95,
        },
    )

    # ---------------------------------------------------------
    # Experimental details
    # ---------------------------------------------------------

    config_text = (
        "FP16 · Batch size 1 · Sequence length 128 · "
        "10 warmup runs · 100 measured forward passes"
    )

    fig.text(
        0.5,
        0.018,
        config_text,
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[0, 0.06, 1, 0.94]
    )

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURES_DIR
        / "profile_forward_latency.png"
    )

    pdf_path = (
        FIGURES_DIR
        / "profile_forward_latency.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 60)
    print("PROFILING FIGURE")
    print("=" * 60)

    for label, value in zip(
        labels,
        mean_latency,
    ):
        print(
            f"{label}: "
            f"{value:.4f} ms mean forward latency"
        )

    print()
    print(
        f"Latency ratio: "
        f"{latency_ratio:.3f}x"
    )

    print(
        f"Latency increase: "
        f"{latency_increase_percent:.2f}%"
    )

    print()
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


def main():
    results = load_results()

    plot_forward_latency(
        results
    )


if __name__ == "__main__":
    main()