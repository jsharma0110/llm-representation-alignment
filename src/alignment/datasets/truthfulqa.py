from datasets import load_dataset


def load_truthfulqa_questions(limit: int | None = None) -> list[str]:
    """
    Load questions from the TruthfulQA generation validation split.

    Args:
        limit: Maximum number of questions to return.
            If None, return all questions.

    Returns:
        List of question strings.
    """
    print("Loading TruthfulQA...")

    dataset = load_dataset(
        "truthful_qa",
        "generation",
    )

    questions = dataset["validation"]["question"]

    if limit is not None:
        questions = questions[:limit]

    questions = list(questions)

    print(f"Loaded {len(questions)} questions.")

    return questions
