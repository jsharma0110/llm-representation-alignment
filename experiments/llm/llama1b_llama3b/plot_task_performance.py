import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path(
    "results/llama1b_llama3b/task_evals"
)

FIGURES_DIR = Path(
    "figures/llama1b_llama3b"
)


FILES = {
    "completion_1b":
        RESULTS_DIR / "completion_llama1b_full.json",

    "completion_3b":
        RESULTS_DIR / "completion_llama3b_full.json",

    "classification_1b":
        RESULTS_DIR / "classification_llama1b_full.json",

    "classification_3b":
        RESULTS_DIR / "classification_llama3b_full.json",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find result file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_results():

    completion_1b = load_json(
        FILES["completion_1b"]
    )

    completion_3b = load_json(
        FILES["completion_3b"]
    )

    classification_1b = load_json(
        FILES["classification_1b"]
    )

    classification_3b = load_json(
        FILES["classification_3b"]
    )

    return {
        "ppl_1b":
            completion_1b["perplexity"],

        "ppl_3b":
            completion_3b["perplexity"],

        "acc_1b":
            classification_1b[
                "accuracy_percent"
            ],

        "acc_3b":
            classification_3b[
                "accuracy_percent"
            ],
    }


def style_axis(ax):
    """Shared formatting for both panels."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="x",
        labelsize=10,
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
    )


def plot_task_performance(results):

    labels = [
        "Llama-3.2-1B",
        "Llama-3.2-3B",
    ]

    perplexities = [
        results["ppl_1b"],
        results["ppl_3b"],
    ]

    accuracies = [
        results["acc_1b"],
        results["acc_3b"],
    ]

    # ---------------------------------------------------------
    # Calculate improvements
    # ---------------------------------------------------------

    ppl_reduction_percent = (
        (
            results["ppl_1b"]
            - results["ppl_3b"]
        )
        / results["ppl_1b"]
        * 100
    )

    accuracy_gain_pp = (
        results["acc_3b"]
        - results["acc_1b"]
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.5),
    )

    # Main title
    fig.suptitle(
        "Larger Model Improves Downstream Performance",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    # Shared directional takeaway
    fig.text(
        0.5,
        0.925,
        "larger model  →  better performance",
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.35",
        fontweight="medium",
    )

    # =========================================================
    # LEFT PANEL: WikiText-2 perplexity
    # =========================================================

    ax = axes[0]

    bars = ax.bar(
        labels,
        perplexities,
        width=0.58,
    )

    ax.set_title(
        "Text Completion",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.text(
        0.5,
        0.94,
        "WikiText-2 · Lower is better",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="0.25",
    )

    ax.set_ylabel(
        "Perplexity",
        fontsize=11,
    )

    ax.set_ylim(
        0,
        max(perplexities) * 1.38,
    )

    style_axis(ax)

    # Values above bars
    for bar, value in zip(
        bars,
        perplexities,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Main takeaway:
    # centered BETWEEN bars and moved downward
    ax.text(
        0.5,
        0.57,
        (
            f"3B: {ppl_reduction_percent:.1f}%\n"
            "lower perplexity"
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.78",
            "alpha": 0.95,
        },
    )

    # =========================================================
    # RIGHT PANEL: SST-2 accuracy
    # =========================================================

    ax = axes[1]

    bars = ax.bar(
        labels,
        accuracies,
        width=0.58,
    )

    ax.set_title(
        "Sentence Classification",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.text(
        0.5,
        0.94,
        "SST-2 · Higher is better",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="0.25",
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=11,
    )

    # Keep zero baseline so the small accuracy gain
    # is not visually exaggerated.
    ax.set_ylim(
        0,
        100,
    )

    style_axis(ax)

    # Values above bars
    for bar, value in zip(
        bars,
        accuracies,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Main takeaway:
    # centered BETWEEN bars and moved downward
    ax.text(
        0.5,
        0.57,
        (
            f"3B: +{accuracy_gain_pp:.2f}\n"
            "percentage points"
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.78",
            "alpha": 0.95,
        },
    )

    # ---------------------------------------------------------
    # Bottom methodology note
    # ---------------------------------------------------------

    fig.text(
        0.5,
        0.018,
        (
            "Llama-3.2-Instruct · "
            "WikiText-2 test set · "
            "SST-2 validation set (872 examples)"
        ),
        ha="center",
        fontsize=9,
        color="0.25",
    )

    # Give the title/subtitle a little more breathing room.
    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.89,
        ]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURES_DIR
        / "task_performance.png"
    )

    pdf_path = (
        FIGURES_DIR
        / "task_performance.pdf"
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

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("=" * 65)
    print("TASK PERFORMANCE FIGURE")
    print("=" * 65)

    print()
    print("WikiText-2:")
    print(
        f"  Llama-3.2-1B: "
        f"{results['ppl_1b']:.4f} perplexity"
    )
    print(
        f"  Llama-3.2-3B: "
        f"{results['ppl_3b']:.4f} perplexity"
    )
    print(
        f"  Reduction: "
        f"{ppl_reduction_percent:.2f}%"
    )

    print()
    print("SST-2:")
    print(
        f"  Llama-3.2-1B: "
        f"{results['acc_1b']:.2f}% accuracy"
    )
    print(
        f"  Llama-3.2-3B: "
        f"{results['acc_3b']:.2f}% accuracy"
    )
    print(
        f"  Gain: "
        f"{accuracy_gain_pp:.2f} percentage points"
    )

    print()
    print(
        f"Saved PNG: {png_path}"
    )
    print(
        f"Saved PDF: {pdf_path}"
    )


def main():

    results = load_results()

    plot_task_performance(
        results
    )


if __name__ == "__main__":
    main()