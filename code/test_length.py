import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from transformers import AutoConfig
import os
import numpy as np

# Device selection: GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Argument parser for CLI input
parser = argparse.ArgumentParser(description="Evaluate a single model on a given dataset")
parser.add_argument("--model", type=str, required=True, help="Model name to evaluate (e.g., llava, linear_0.25)")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., okvqa)")
args = parser.parse_args()

# Load model & processor
model_base_path = "/media/chagan/yaga9887"
data_base_path = "/home/yaga9887/toy_data"

model_name = args.model
dataset_name = args.dataset

model_path = os.path.join(model_base_path, model_name)
dataset_path = os.path.join(data_base_path, dataset_name, f"{dataset_name}_toy.json")

# Load model and processor
if model_name.startswith("linear"):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if hasattr(config, "text_config"):
        for key, value in config.text_config.to_dict().items():
            if not hasattr(config, key):
                setattr(config, key, value)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True)

elif model_name.startswith("llava"):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True)

else:  # deepseek
    processor = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

model.to(dtype=torch.float16 if "llava" in model_name else torch.bfloat16, device="cuda")
model.eval()

# Load dataset
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []
gen_lens = []  # 🔍 用于统计生成长度

# Inference loop
for item in data:
    question = item['question']
    answer = item['answer']
    image_path = item['image_path']

    # Create text_input（including choices）
    if 'choices' in item and isinstance(item['choices'], list):
        choices_text = "\nChoices:\n" + "\n".join([f"{i}. {choice}" for i, choice in enumerate(item['choices'])])
    else:
        choices_text = ""

    text_input = f"Question: {question}{choices_text}"

    result_entry = {
        "question": question,
        "answer": answer,
        "image_path": image_path,
        f"{model_name}_result": ""
    }

    if model_name.startswith("deepseek"):
        conversation = [{"role": "user", "content": text_input}]
        prompt = text_input + "<think>\n"
        inputs = processor(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=processor.eos_token_id,
                eos_token_id=processor.eos_token_id
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        gen_len = generated_ids.shape[0]
        print(f"[DEBUG] Generated token count: {gen_len}")
        gen_lens.append(gen_len)

        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        result_entry[f"{model_name}_result"] = decoded

    elif model_name.startswith("llava") or model_name.startswith("linear"):
        image = Image.open(image_path).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "text", "text": text_input}, {"type": "image"}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        additional_prompt = "Please think step-by-step and explain your reasoning before the final answer."
        prompt += additional_prompt + "<think>\n"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,

                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        gen_len = generated_ids.shape[0]
        print(f"[DEBUG] Generated token count: {gen_len}")
        gen_lens.append(gen_len)
        decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        result_entry[f"{model_name}_result"] = decoded

    results.append(result_entry)

# Output to JSON file
output_file = f"results_{model_name}_{dataset_name}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 📊 输出生成长度的统计信息
if gen_lens:
    print("\n[STATS] Generated token length summary:")
    print("  Count:", len(gen_lens))
    print("  Max token length: ", int(np.max(gen_lens)))
    print("  95th percentile:  ", int(np.percentile(gen_lens, 95)))
    print("  Mean token length:", round(np.mean(gen_lens), 2))
    print("  Std dev:           ", round(np.std(gen_lens), 2))

print(f"[INFO] Evaluation completed. Results saved to {output_file}")

