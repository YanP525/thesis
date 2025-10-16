import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
from tqdm import tqdm
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

parser = argparse.ArgumentParser(description="Evaluate model on dataset with few-shot and accuracy calculation")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., llava, linear_0.25)")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (folder name)")
args = parser.parse_args()

model_base_path = "/media/chagan/yaga9887"
data_base_path = "/home/yaga9887/toy_data"

model_name = args.model
dataset_name = args.dataset

model_path = os.path.join(model_base_path, model_name)
dataset_path = os.path.join(data_base_path, dataset_name, f"{dataset_name}_toy.json")

if model_name.startswith("deepseek"):
    processor = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
elif model_name.startswith("llava") or model_name.startswith("linear"):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True)

model.to(dtype=torch.float16 if "llava" in model_name else torch.bfloat16, device=device)
model.eval()

# Few-shot examples
fewshot_examples = [
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "What is this desk used for?Please answer directly with only the letter of the correct option and nothing else.",
            "answer": [
                "working",
                "working",
                "working",
                "working",
                "computing",
                "computing",
                "work",
                "work",
                "office",
                "office"
            ]
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Okay, so I'm trying to figure out what this desk is used for. The question is asking me to answer directly with only the letter of the correct option. I don't have the options in front of me, but I'll try to think it through. \n\nFirst, I need to visualize the desk. Desks come in various types: some are for office work, others for home use, maybe even for gaming or studying. Since the question is asking about what it's used for, I should consider the common purposes of a desk. \n\nA standard desk is typically where people work, so maybe it's for office work, like writing, reading, or using a computer. But it could also be a desk in a home office or a student's study space. Alternatively, a desk might be used for other activities like eating, but that's less common. \n\nWait, but the user mentioned \"Please answer directly with only the letter of the correct option.\" So they might be referring to a multiple-choice question where each option is labeled with a letter. Since I don't have the options, I need to infer based on common uses.\n\nIf I think about the most common uses, office work is probably the top use. So the answer is likely B for office work. But I should consider other possibilities. Maybe it's for homework or studying, which could be similar. Or perhaps it's for a specific activity like gaming, but that's less likely unless it's a gaming desk. \n\nI'm leaning towards office work because that's the most general use for a desk. So I'll go with B.\n</think>\n\nB"}
        ]
    },
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Is this at a salt water beach or a lake?Please answer directly with only the letter of the correct option and nothing else.",
            "answer": [
                "salt water beach",
                "salt water beach",
                "salt water beach",
                "salt water beach",
                "salt water",
                "salt water",
                "lake",
                "lake",
                "beach",
                "beach"
            ]
            }
        ]
    },

    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Okay, so I need to figure out whether the location in question is a saltwater beach or a lake. The user hasn't provided any specific details, so I'll have to make an educated guess based on the information given. Let me think about the typical characteristics of each.\n\nFirst, saltwater beaches are usually near oceans or large bodies of saltwater like the sea. They have the ocean's influence, so the water is salty, and there's often a soft, sandy shoreline with maybe some rocks or coral. The waves are usually stronger and have a certain texture. The environment around them might have vegetation like mangroves or dunes.\n\nOn the other hand, lakes are freshwater bodies, which can be natural like lakes or man-made like reservoirs. Freshwater lakes don't have the same salt content as the ocean, so the water is softer. The shores of lakes are often more defined, with cliffs, trees, or grassy areas. The water might be calmer because it's not influenced by the tides.\n\nSince the question is asking about a specific location, I need to think about what clues might indicate saltwater vs. freshwater. Without specific details, it's tricky, but perhaps the mention of certain features like tidal patterns, the presence of marine life, or the type of water can help. For example, if there's mention of tide, that's a strong indicator for a saltwater beach.\n\nIf the location has a white, sandy beach with wave breaks, that's likely a saltwater beach. If it's surrounded by trees, has a calm surface, and maybe a boat launch, that might lean towards a lake.\n\nBut since I don't have specifics, I might have to go with the most common association. Often, when people refer to a beach without more context, they mean a saltwater beach. So, I'll go with that.\n</think>\n\nsaltwater beach"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Why might someone go to this place?Please answer directly with only the letter of the correct option and nothing else.",
            "answer": [
                "shopping",
                "shopping",
                "shopping",
                "shopping",
                "nyc",
                "nyc",
                "shop",
                "shop",
                "business",
                "business"
            ]}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Okay, so I have this question: \"Why might someone go to this place?\" And I need to answer directly with just the letter of the correct option. Hmm, but wait, the user hasn't provided the options. Maybe they're referring to a specific place, but it's not mentioned here. Did I miss something? Let me think. Perhaps the \"this place\" is a reference to a common location, like a park, a museum, a beach, a city, or maybe a specific store or event. Without more context, it's a bit tricky. \n\nWait, maybe the user is referring to a previous conversation or a context I'm not aware of. Oh, I should probably ask for more details. But the user wants me to answer directly with just the letter, so maybe they expect me to infer based on common reasons. Let me consider the most common reasons someone might go to a place. It could be for relaxation, tourism, work, shopping, visiting someone, a conference, or maybe a specific event like a concert or festival. \n\nIf I had to guess, the most generic and common reasons are tourism or visiting a place for leisure. So maybe the answer is A. But without the options, I'm not sure. Alternatively, if I think about places like parks, people go for relaxation or exercise, so maybe B. Or if it's a business-related, then C. I'm a bit confused. I think the safest answer is to ask for more details, but since the user wants a direct letter, I'll go with the most common, which is probably A. tourism.\n</think>\n\nA"}
        ]
    }


    # {
    #     "role": "user",
    #     "content": [{"type": "text", "text": "Q: Is the sun a star?"}]
    # },
    # {
    #     "role": "assistant",
    #     "content": [{"type": "text", "text": "<think>\nYes, the sun is a star at the center of our solar system.\n</think>\nAnswer: Yes"}]
    # },
    # {
    #     "role": "user",
    #     "content": [{"type": "text", "text": "Q: How many continents are there on Earth?"}]
    # },
    # {
    #     "role": "assistant",
    #     "content": [{"type": "text", "text": "<think>\nThere are seven continents on Earth.\n</think>\nAnswer: 7"}]
    # },
    #         {
    #     "role": "user",
    #     "content": [{"type": "text", "text": "Q: Why would someone visit a busy market street?"}]
    # },
    # {
    #     "role": "assistant",
    #     "content": [{"type": "text", "text": "<think>\nMarket streets are typically places where people go to shop for food, clothes, or household items.\n</think>\nAnswer: shopping"}]
    # },
    #         {
    #     "role": "user",
    #     "content": [{"type": "text", "text": "Q: What is a common reason for going to the city center on a weekday?"}]
    # },
    # {
    #     "role": "assistant",
    #     "content": [{"type": "text", "text": "<think>\nPeople often go to the city center for work, business meetings, or errands.\n</think>\nAnswer: business"}]
    # }
]

import re

def extract_pred_answer(decoded, gt_answer):
    """
    提取预测答案，支持 <think> 标签，也支持无标签的输出。
    """
    if "</think>" in decoded:
        after_think = decoded.split("</think>")[-1].strip()
        candidate_lines = after_think.splitlines()
    else:
        candidate_lines = decoded.strip().splitlines()

    pred_raw = ""
    for line in reversed(candidate_lines):
        line = line.strip()
        if line and not line.lower().startswith("<think>"):
            pred_raw = line
            break

    match = re.search(r"\b([A-F]|YES|NO|\d+)\b", pred_raw, re.IGNORECASE)
    pred = match.group(1).strip() if match else pred_raw.strip()

    gt_clean = gt_answer.strip()
    try:
        pred_norm = str(int(pred))
        gt_norm = str(int(gt_clean))
    except ValueError:
        pred_norm = pred.upper()
        gt_norm = gt_clean.upper()

    is_correct = pred_norm == gt_norm

    return pred, is_correct

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f) 

results = []
correct_count = 0

for item in tqdm(data):
    question = item['question']
    answer = item['answer']
    choices = item.get('choices', [])
    image_path = item['image_path']

    if choices:
        choices_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
        text_input = f"Q: {question}\n{choices_text}"
    else:
        text_input = f"Q: {question}"

    if model_name.startswith("deepseek"):
        # deepseek分支不变（省略）

        conversation = [{"role": "user", "content": text_input + " <think>\n"}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=512)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()

        pred, is_correct = extract_pred_answer(decoded, answer)
        result_entry = {
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "choices": choices,
            f"{model_name}_result": decoded,
            "correct": is_correct,
        }
        if is_correct:
            correct_count += 1
        results.append(result_entry)

    else:  # llava / linear
        image = Image.open(image_path).convert("RGB")
        conversation = fewshot_examples + [{
            "role": "user",
            "content": [{"type": "text", "text": text_input + "\n<think>\n"}, {"type": "image"}]
        }]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            # 第一步：生成 CoT 思考内容，直到 </think> (没有stopcriteria示例，假设 max_new_tokens 控制长度)
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=1024
            )
        prompt_len = inputs["input_ids"].shape[-1]
        cot_ids = output_ids[0][prompt_len:]
        cot_text = processor.decode(cot_ids, skip_special_tokens=True).strip()

        # 拼接输入：原输入 + 生成的 CoT 内容
        new_input_ids = torch.cat([inputs["input_ids"][0], cot_ids], dim=0).unsqueeze(0).to(device)
        new_attention_mask = torch.ones_like(new_input_ids)

        with torch.no_grad():
            # 单步推理，得到下一个token的logits
            outputs = model(input_ids=new_input_ids, attention_mask=new_attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            # 取概率最高token作为答案token
            next_token_id = torch.argmax(next_token_logits, dim=-1)

        # 将答案token拼接到 CoT 后面，解码完整文本（CoT + 答案）
        full_ids = torch.cat([new_input_ids[0], next_token_id], dim=0)
        decoded_full = processor.decode(full_ids, skip_special_tokens=True).strip()

        # 提取预测答案（答案部分是 next_token_id 解码的那部分）
        # 这里直接用 next_token_id 解码更准确
        pred = processor.decode(next_token_id, skip_special_tokens=True).strip()

        gt_answer = answer
        # 标准化比较
        try:
            pred_norm = str(int(pred))
            gt_norm = str(int(gt_answer.strip()))
        except ValueError:
            pred_norm = pred.upper()
            gt_norm = gt_answer.strip().upper()
        is_correct = pred_norm == gt_norm

        result_entry = {
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "choices": choices,
            f"{model_name}_result": decoded_full,
            "correct": is_correct,
            "cot_text": cot_text,
            "pred_answer": pred,
        }
        if is_correct:
            correct_count += 1

        results.append(result_entry)

print(f"[INFO] Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.4f}")

# 保存结果
output_file = f"results_{model_name}_{dataset_name}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[INFO] Results saved to {output_file}")

