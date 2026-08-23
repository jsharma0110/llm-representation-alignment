from pathlib import Path

import numpy as np

from alignment.similarity import compute_cka_matrix


TINYLLAMA_DIR = Path("results/hidden_states/tinyllama")
QWEN_DIR = Path("results/hidden_states/qwen")

OUTPUT_DIR = Path("results/tinyllama_qwen")


def main():
    similarity = compute_cka_matrix(
        model_a_dir=TINYLLAMA_DIR,
        model_b_dir=QWEN_DIR,
        standardize=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUTPUT_DIR / "cka_matrix.npy",
        similarity,
    )

    np.savetxt(
        OUTPUT_DIR / "cka_matrix.csv",
        similarity,
        delimiter=",",
        fmt="%.6f",
    )

    print(
        f"\nSaved CKA matrix to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
