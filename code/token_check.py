from transformers import AutoProcessor
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="The name of the model folder under /media/chagan/yaga9887/")
    args = parser.parse_args()

    base_path = "/media/chagan/yaga9887"
    model_path = os.path.join(base_path, args.model_name)

    tokenizer = AutoProcessor.from_pretrained(model_path).tokenizer

    print("Tokens from ID 128000 onwards (excluding <|reserved_...>):")
    for token_id in range(128000, len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token.startswith("<|reserved_"):
            continue
        print(f"ID {token_id}: {token}")

if __name__ == "__main__":
    main()

