import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import argparse
import torch.nn as nn
import os
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# Load LLaVA and DeepSeek models and their tokenizers
def load_models():
    print("Loading LLaVA model...")
    llava_model = AutoModelForImageTextToText.from_pretrained("/media/chagan/yaga9887/llava_bak")
    llava_processor = AutoProcessor.from_pretrained("/media/chagan/yaga9887/llava_bak")
    llava_tokenizer = llava_processor.tokenizer
    print("LLaVA model loaded successfully.")
    print("Loading DeepSeek model...")
    deepseek_model = AutoModelForCausalLM.from_pretrained("/media/chagan/yaga9887/deepseek")
    deepseek_tokenizer = AutoTokenizer.from_pretrained("/media/chagan/yaga9887/deepseek")
    print("DeepSeek model loaded successfully.")

    return llava_model, llava_tokenizer, deepseek_model, deepseek_tokenizer


def merge_backbone(llava_model, alpha):
    # Model paths
    llava_model_paths = [
        "/media/chagan/yaga9887/llava/model-00001-of-00004.safetensors",
        "/media/chagan/yaga9887/llava/model-00002-of-00004.safetensors",
        "/media/chagan/yaga9887/llava/model-00003-of-00004.safetensors",
        "/media/chagan/yaga9887/llava/model-00004-of-00004.safetensors"
    ]
    deepseek_model_paths = [
        "/media/chagan/yaga9887/deepseek/model-00001-of-000002.safetensors",
        "/media/chagan/yaga9887/deepseek/model-00002-of-000002.safetensors"
    ]

    # Load model parameters
    llava_params = {}
    for path in llava_model_paths:
        llava_params.update(load_file(path))

    deepseek_params = {}
    for path in deepseek_model_paths:
        deepseek_params.update(load_file(path))

    # Filter LLaVA language model parameters
    filtered_llava_params = {}
    for key, value in llava_params.items():
        if key.startswith("language_model"):
            if key == "language_model.model.embed_tokens.weight" or key == "language_model.lm_head.weight":
                continue
            else:
                filtered_llava_params[key] = value

    # Adjust DeepSeek parameters by adding "language_model." prefix
    adjusted_deepseek_params = {}
    for key, value in deepseek_params.items():
        if key.startswith("model"):
            if key == "model.embed_tokens.weight" or key == "lm_head.weight":
                continue
            else:
                new_key = "language_model." + key
                adjusted_deepseek_params[new_key] = value

    # Ensure the keys of both parameter dictionaries are consistent
    merged_params = {}
    for key in tqdm(filtered_llava_params, desc="Merging parameters", unit="param"):
        if key in adjusted_deepseek_params:
            # Use linear weighted merging
            merged_params[key] = (adjusted_deepseek_params[key] * alpha +
                                  filtered_llava_params[key] * (1 - alpha))
        else:
            print(f"Warning: {key} is missing from DeepSeek parameters.")

    # Load the merged parameters into the model
    llava_model.load_state_dict(merged_params, strict=False)
    print(f"Backbone merged")
    return llava_model

# Merge two tokenizers and update vocabulary
def merge_tokenizers(llava_tokenizer, deepseek_tokenizer, alpha):
    print("Merging tokenizers...")

    # Get vocabularies from both tokenizers
    llava_vocab = llava_tokenizer.get_vocab()
    deepseek_vocab = deepseek_tokenizer.get_vocab()
    print(f"LLaVA vocab size: {len(llava_vocab)}")
    print(f"DeepSeek vocab size: {len(deepseek_vocab)}")

    # Find unique tokens
    # tokens_only_in_llava = set(llava_vocab) - set(deepseek_vocab)
    tokens_only_in_deepseek = set(deepseek_vocab) - set(llava_vocab)

    # Merge vocabularies
    merged_vocab = llava_vocab.copy()
    for token in tokens_only_in_deepseek:
        merged_vocab[token] = len(merged_vocab)

    print(f"Total tokens after merge: {len(merged_vocab)}")

    return merged_vocab

# Merge the embedding layers of the two models
def merge_embeddings(llava_model, deepseek_model, llava_tokenizer, deepseek_tokenizer, merged_vocab, alpha):
    print("Merging model embeddings...")

    # Get the original embedding layers
    llava_embed = llava_model.get_input_embeddings()
    deepseek_embed = deepseek_model.get_input_embeddings()

    # Get embedding dimension
    embed_dim = llava_embed.embedding_dim
    print(f"Embedding dimension: {embed_dim}")

    # Create merged embedding matrix
    vocab_size = len(merged_vocab)
    print(f"Total merged vocab size: {vocab_size}")
    merged_embed = torch.zeros((vocab_size, embed_dim))

    for token, new_idx in merged_vocab.items():
        try:
            llava_id = llava_tokenizer.convert_tokens_to_ids(token)
        except:
            llava_id = None

        try:
            deepseek_id = deepseek_tokenizer.convert_tokens_to_ids(token)
        except:
            deepseek_id = None

        llava_vec = llava_embed.weight[llava_id] if llava_id is not None and llava_id < llava_embed.weight.shape[0] else torch.zeros(embed_dim)
        deepseek_vec = deepseek_embed.weight[deepseek_id] if deepseek_id is not None and deepseek_id < deepseek_embed.weight.shape[0] else torch.zeros(embed_dim)

        merged_vec = alpha * deepseek_vec + (1 - alpha) * llava_vec
        merged_embed[new_idx] = merged_vec

    # Padding to 128320
    target_vocab_size = 128320
    if merged_embed.shape[0] < target_vocab_size:
        pad_tensor = torch.zeros((target_vocab_size - merged_embed.shape[0], embed_dim))
        merged_embed = torch.cat([merged_embed, pad_tensor], dim=0)


    return merged_embed

# Merge lm_head weights of the two models
def merge_lm_head(llava_model, deepseek_model, merged_vocab, llava_tokenizer, deepseek_tokenizer, alpha):
    print("Merging lm_head weights...")

    llava_lm_head = llava_model.language_model.lm_head.weight
    deepseek_lm_head = deepseek_model.lm_head.weight

    vocab_size = len(merged_vocab)
    hidden_size = llava_lm_head.shape[1]  # 4096

    merged_lm_head = torch.zeros((vocab_size, hidden_size))

    for token, idx in merged_vocab.items():
        llava_id = llava_tokenizer.convert_tokens_to_ids(token)
        deepseek_id = deepseek_tokenizer.convert_tokens_to_ids(token)

        in_llava = llava_id is not None and 0 <= llava_id < llava_lm_head.shape[0]
        in_deepseek = deepseek_id is not None and 0 <= deepseek_id < deepseek_lm_head.shape[0]

        llava_vec = llava_lm_head[llava_id] if in_llava else torch.zeros(hidden_size)
        deepseek_vec = deepseek_lm_head[deepseek_id] if in_deepseek else torch.zeros(hidden_size)
        merged_lm_head[idx] = alpha * deepseek_vec + (1 - alpha) * llava_vec

    # padding
    target_vocab_size = 128320
    if merged_lm_head.shape[0] < target_vocab_size:
        pad_tensor = torch.zeros((target_vocab_size - merged_lm_head.shape[0], hidden_size))
        merged_lm_head = torch.cat([merged_lm_head, pad_tensor], dim=0)

    return merged_lm_head

def create_new_tokenizer(merged_vocab, base_tokenizer):
    existing_vocab = base_tokenizer.get_vocab()
    new_tokens = [tok for tok in merged_vocab if tok not in existing_vocab]
    print(f"Adding {len(new_tokens)} new tokens to tokenizer...")

    # Add new tokens
    base_tokenizer.add_tokens(new_tokens)
    return base_tokenizer


# Update the model with merged weights and save
def save_model_and_tokenizer(merged_vocab, merged_embed, merged_lm_head, backbone_merged_model, new_tokenizer, alpha, output_dir):
    # llava_processor = AutoProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    # llava_tokenizer = llava_processor.tokenizer
    model_dir = os.path.join(output_dir, f"linear_{alpha}")
    # print(f"Saving merged model to {output_dir} with alpha={alpha}...")

    os.makedirs(model_dir, exist_ok=True)

    # Create new mbedding
    new_embed_layer = nn.Embedding.from_pretrained(merged_embed)

    # Set backbone-merged model's embed_tokens to the merged embedding layer
    backbone_merged_model.language_model.model.embed_tokens = new_embed_layer
    merged_lm_head_param = torch.nn.Parameter(merged_lm_head)
    backbone_merged_model.language_model.lm_head.weight = merged_lm_head_param
    backbone_merged_model.config.vocab_size = len(merged_vocab)

    merged_model = backbone_merged_model.to(torch.bfloat16)
    try:
        merged_model.save_pretrained(model_dir, safe_serialization=True)
        print(f"Model saved successfully to {model_dir}")

        # merged_model.config.to_json_file(os.path.join(model_dir, 'config.json'))
        # print(f"Config file saved successfully to {model_dir}")

    except Exception as e:
        print(f"Error while saving the model: {e}")

    new_tokenizer.save_pretrained(model_dir)
    print(f"Tokenizer saved successfully to {model_dir}")

    # Make sure index.json is saved
    try:
        safetensors_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(safetensors_path):
            print(f"Safetensors index.json already exists at {safetensors_path}")
        else:
            print(f"Warning: Safetensors index.json does not exist.")
    except Exception as e:
        print(f"Error while handling safetensors: {e}")


def main(alpha, output_dir):
    llava_model, llava_tokenizer, deepseek_model, deepseek_tokenizer = load_models()

    backbone_merged_model = merge_backbone(llava_model, alpha)
    # Merge tokenizers
    merged_vocab = merge_tokenizers(llava_tokenizer, deepseek_tokenizer, alpha)
    # llava_vocab = llava_tokenizer.get_vocab()

    # Merge model embeddings and lm_head
    merged_embed = merge_embeddings(llava_model, deepseek_model, llava_tokenizer, deepseek_tokenizer, merged_vocab, alpha)
    merged_lm_head = merge_lm_head(llava_model, deepseek_model, merged_vocab, llava_tokenizer, deepseek_tokenizer, alpha)

    new_tokenizer = create_new_tokenizer(merged_vocab, llava_tokenizer)
    print("Tokenizer vocab size:", new_tokenizer.vocab_size)

    # new_tokenizer.save_pretrained(output_dir)
    # print(f"New tokenizer saved to {output_dir}")
    save_model_and_tokenizer(merged_vocab, merged_embed, merged_lm_head, backbone_merged_model, new_tokenizer, alpha, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True, help="The blending factor for the models (0 to 1).")
    parser.add_argument('--output_dir', type=str, default='/media/chagan/yaga9887', help='Directory to save merged model')
    args = parser.parse_args()

    main(args.alpha, args.output_dir)
