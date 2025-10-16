import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse

def build_llava_prompt(question: str) -> str:
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged model folder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    print("Model and tokenizer loaded. You can type questions now. Type 'exit' to quit.")

    while True:
        query = input("Question: ").strip()
        if query.lower() == "exit":
            break

        prompt = build_llava_prompt(query)
        inputs = processor(text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
            )

        output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # 去掉输入的prompt，只保留模型回答
        answer = output_text[len(prompt):].strip()
        print("Answer:", answer)

if __name__ == "__main__":
    main()

