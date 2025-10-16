import torch
import sys
import os
from transformers import (
    AutoProcessor, AutoTokenizer,
    AutoModelForCausalLM, AutoModelForImageTextToText
)

def load_model_and_tokenizer(model_name, base_path="/media/chagan/yaga9887"):
    model_path = os.path.join(base_path, model_name)

    if model_name.startswith("llava") or model_name.startswith("linear"):
        tokenizer = AutoProcessor.from_pretrained(model_path, local_files_only=True).tokenizer
        model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.startswith("deepseek"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unrecognized model type for: {model_name}")

    return tokenizer, model, model_path

def main():
    if len(sys.argv) != 3:
        print("Usage: python print_unq_params.py <model1_name> <model2_name>")
        sys.exit(1)

    model1_name, model2_name = sys.argv[1], sys.argv[2]

    # Load models and tokenizers
    tokenizer1, model1, path1 = load_model_and_tokenizer(model1_name)
    tokenizer2, model2, path2 = load_model_and_tokenizer(model2_name)

    embed1 = model1.get_input_embeddings().weight.data
    embed2 = model2.get_input_embeddings().weight.data

    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    id2token1 = {v: k for k, v in vocab1.items()}
    id2token2 = {v: k for k, v in vocab2.items()}

    # Token ID range
    start_id = 127990
    end_id = 128267

    output_file = f"/home/yaga9887/token_embeddings_comparison_{model1_name}_vs_{model2_name}.txt"
    with open(output_file, "w") as f:
        f.write(f"Token Embedding Comparison (IDs {start_id}-{end_id}):\n")
        f.write(f"{'Token ID':<10} | {model1_name:<30} | {model2_name:<30} | {'Shape 1':<15} | {'Shape 2':<15}\n")
        f.write("-" * 120 + "\n")

        for token_id in range(start_id, end_id + 1):
            token1 = id2token1.get(token_id, "N/A")
            token2 = id2token2.get(token_id, "N/A")

            try:
                vec1 = embed1[token_id]
                shape1 = str(vec1.shape)
                vals1 = ", ".join([f"{v:.4f}" for v in vec1[:5]])
            except IndexError:
                shape1 = "N/A"
                vals1 = "N/A"

            try:
                vec2 = embed2[token_id]
                shape2 = str(vec2.shape)
                vals2 = ", ".join([f"{v:.4f}" for v in vec2[:5]])
            except IndexError:
                shape2 = "N/A"
                vals2 = "N/A"

            f.write(f"{token_id:<10} | {token1:<30} | {token2:<30} | {shape1:<15} | {shape2:<15}\n")
            f.write(f"{'':<10} | {model1_name} vals: {vals1}\n")
            f.write(f"{'':<10} | {model2_name} vals: {vals2}\n\n")

    print(f"Embedding details written to {output_file}")

if __name__ == "__main__":
    main()
