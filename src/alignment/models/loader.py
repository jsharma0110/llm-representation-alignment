from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name: str):
    """
    Load a causal language model and its tokenizer.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    return model, tokenizer
