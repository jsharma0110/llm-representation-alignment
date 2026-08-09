from pathlib import Path

import numpy as np

from alignment.similarity import compute_cka_matrix


LLAMA1B_DIR = Path("results/llama1b")
LLAMA3B_DIR = Path("results/llama3b")

OUTPUT_DIR = Path("results/llama1b_llama3b")


def main():
    similarity = compute_cka_matrix(
        model_a_dir=LLAMA1B_DIR,
        model_b_dir=LLAMA3B_DIR,
        standardize=True,
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
