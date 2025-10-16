import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import os
from tqdm import tqdm
import re
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

parser = argparse.ArgumentParser(description="Evaluate model on dataset with few-shot and accuracy calculation")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., llava, linear_0.25)")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (folder name)")
args = parser.parse_args()

model_base_path = "/media/chagan/yaga9887"
data_base_path = "/home/yaga9887/toy_data"
#data_base_path = "/media/chagan/yaga9887/dataset"

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
            {"type": "text", "text": "You invest $1 at a 5% interest rate for 3 years. What is the present value of that $1?\n0. 0.8638\n1. 1\n2. 14\n3. 10\nPlease answer directly with only the letter of the correct option and nothing else.",
            "answer": "0"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "To determine the present value of \\$1 invested at a 5% annual interest rate for 3 years, I'll use the present value formula for compound interest.\n\nFirst, I'll identify the variables: the principal (P) is \\$1, the annual interest rate (r) is 5%, and the number of years (n) is 3.\n\nNext, I'll convert the interest rate to a decimal by dividing 5% by 100, which gives 0.05.\n\nThen, I'll apply the present value formula: PV = P × (1 + r)^n.\n\nPlugging in the values: PV = 1 × (1 + 0.05)^3.\n\nCalculating (1.05)^3 equals approximately 1.157625.\n\nFinally, multiplying by the principal: PV ≈ 1 × 1.157625 = 1.157625.\n\nComparing this to the provided options, the closest match is option 1.\n</think>\n\nTo determine the present value of \\$1 invested at a 5% annual interest rate for 3 years, we'll use the **present value formula** for compound interest:\n\n\\[\nPV = P \\times (1 + r)^n\n\\]\n\nWhere:\n- \\( PV \\) is the present value (what we want to find),\n- \\( P \\) is the principal amount (\\$1),\n- \\( r \\) is the annual interest rate (5% or 0.05),\n- \\( n \\) is the number of years (3).\n\nPlugging in the values:\n\n\\[\nPV = 1 \\times (1 + 0.05)^3\n\\]\n\n\\[\nPV = 1 \\times (1.05)^3\n\\]\n\nCalculating \\( (1.05)^3 \\):\n\n\\[\n1.05 \\times 1.05 = 1.1025\n\\]\n\\[\n1.1025 \\times 1.05 = 1.157625\n\\]\n\nSo,\n\n\\[\nPV \\approx 1.157625\n\\]\n\nComparing this to the provided options:\n\n0. 0.8638  \n1. 1  \n2. 14  \n3. 10  \n\nThe closest match is **1**.\n</think>\n\n1"}
        ]
    },
    
    {
       "role": "user",
       "content": [
           {
           "type": "text",
           "text": "A small project includes several tasks with different durations. The longest path through the tasks takes 26 days. What is the total project completion time?\n0. 23\n1. 25\n2. 26\n3. 30\nPlease answer directly with only the letter of the correct option and nothing else.",
          "answer": "2"
           }
       ]
    },
    {
       "role": "assistant",
       "content": [
           {"type": "text", "text": "To determine the total project completion time, I need to understand the nature of the tasks and their durations. However, the problem only mentions that the longest path through the tasks takes 26 days but doesn't provide specific details about the individual task durations or the number of tasks. Without this information, it's impossible to calculate the total project completion time accurately.\n\nTherefore, with the given data, I cannot determine the correct answer.\nTo determine the total project completion time, we need more information about the individual tasks and their durations. The problem only mentions that the longest path through the tasks takes 26 days, but it doesn't provide specific details about the number of tasks or their individual durations. Without this information, it's impossible to calculate the total project completion time accurately.\n</think>\n\n2"}
       ]
    },
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Seven students each take 3 different exams. No student can take more than one exam per day. What is the minimum number of days needed to schedule all exams?\n0. 3\n1. 4\n2. 5\n3. 6\nPlease answer directly with only the letter of the correct option and nothing else.",
            "answer": "1"
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text":  "To determine the minimum number of days needed to schedule all exams, we start by noting that there are 7 students and each must take 3 different exams. This means there are a total of 21 exams to be scheduled.\n\nEach day, a student can take only one exam. Therefore, each day can accommodate 7 exams (one for each student). \n\nBy calculating the total number of exams divided by the number of exams that can be held each day, we get 21 exams ÷ 7 exams per day = 3 days.\n\nThus, the minimum number of days required is 3.\n</think>\n\nTo determine the minimum number of days needed to schedule all exams, let's analyze the problem step by step.\n\n**Given:**\n- **Number of students:** 7\n- **Exams per student:** 3\n- **Exams per day per student:** 1\n\n**Total exams to schedule:**  \nSince each of the 7 students takes 3 exams, the total number of exams is:\n\\[ 7 \\text{ students} \\times 3 \\text{ exams/student} = 21 \\text{ exams} \\]\n\n**Exams per day:**\n- Each day, a student can take **1 exam**.\n- Therefore, each day can accommodate **7 exams** (one for each student).\n\n**Number of days required:**\n\\[ \\frac{21 \\text{ exams}}{7 \\text{ exams/day}} = 3\nComparing this to the provided options:\n\\n0. 3\n1. 4\n2. 5\n3. 6\n\nThe closest match is **0**.\n</think>\n\n0"}
        ]
    }
]


class StopAfterThinkAnswer(StoppingCriteria):
    def __init__(self, tokenizer, max_tokens_after_think=30):
        self.tokenizer = tokenizer
        self.after_think = False  # Flag to mark if </think> has appeared
        self.accumulated_tokens = []  # Tokens generated after </think>
        self.max_tokens_after_think = max_tokens_after_think
        self.newline_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
        self.fired = False  # Flag to mark if stopping criteria has been triggered

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.fired:
            return True

        new_token_id = input_ids[0, -1].item()
        self.accumulated_tokens.append(new_token_id)

        # Decode the accumulated tokens, skipping special tokens
        current_text = self.tokenizer.decode(self.accumulated_tokens, skip_special_tokens=True)

        if not self.after_think:
            # Check if </think> token sequence appeared, switch flag and reset tokens
            if "</think>" in current_text:
                self.after_think = True
                self.accumulated_tokens = []
        else:
            # After </think>, look for newline token to identify answer lines
            if self.newline_token_id in self.accumulated_tokens:
                answer_str = self.tokenizer.decode(self.accumulated_tokens, skip_special_tokens=True).strip()
                lines = answer_str.split("\n")
                for line in lines:
                    if line.strip():  # Stop generation once a non-empty line is found
                        self.fired = True
                        return True
            # Also stop if max tokens after </think> exceeded to prevent runaway generation
            if len(self.accumulated_tokens) >= self.max_tokens_after_think:
                self.fired = True
                return True

        return False #not stopping

def extract_pred_answer(decoded: str, gt_answer: str):
    """
    Extract the predicted answer from the full decoded text.
    - If </think> is present, take the first non-empty line after it.
    - Otherwise, take the last non-empty line of the output.
    Also compares the prediction with the ground truth for correctness.
    """
    if "</think>" in decoded:
        segment = decoded.split("</think>", 1)[1]
    else:
        segment = decoded

    lines = [line.strip() for line in segment.strip().splitlines() if line.strip()]
    pred_raw = lines[0] if lines else ""

    pred_clean = str(pred_raw).lower().strip().rstrip(".")
    gt_clean = str(gt_answer).lower().strip().rstrip(".")
    is_correct = pred_clean == gt_clean

    return pred_raw, is_correct


def build_prompt(question, choices):
    # for datasets which take indices as ground truth
    indexed_choices = [f"{i}. {choice}" for i, choice in enumerate(choices)]
    # choices = ast.literal_eval(choices) #make sure "choices" is a list not a string
    # indexed_choices = [f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)]
    choice_str = "\n".join(indexed_choices)
    
    prompt = f"{question}\n{choice_str}\nPlease answer directly with only the letter of the correct option and nothing else."
    return prompt

def convert_to_deepseek_format(fewshot_examples):
    converted = []
    for msg in fewshot_examples:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            text_parts = [c["text"] for c in content if "text" in c]
            content = "\n".join(text_parts)
        converted.append({"role": role, "content": content})
    return converted


# read file
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []
correct_count = 0

for item in tqdm(data):
    question = item['question']
    answer = item['answer']
    choices = item.get('choices', [])
    image_path = item.get('image_path', None)
    subject = item['subject']

    #construct different prompt accoring to question types
    # if choices:
    #    #choices_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
    #    text_input = f"Q: {question}\n{choices}\nPlease select one choice and write out the full answer and nothing else."
    # else:
    #    text_input = f"Q: {question}\nPlease answer the question providing the final value and nothing else."
    text_input = build_prompt(question, choices)

    if model_name.startswith("deepseek"):
        conversation = [{"role": "user", "content": text_input + "\n<think>\n"}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1024)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        gen_len = generated_ids.shape[0]
        print(f"Generated token count: {gen_len}")
        decoded = processor.decode(generated_ids, skip_special_tokens=True)

    else:  # llava / linear
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = None
        # content = [{"type": "image"} for _ in images]  
        # content.append({"type": "text", "text": text_input + "\n<think>\n"})  
        # if question_type == "multi_choice":
        #     selected_fewshot = fewshot_examples_typeA
        # else:
        #     selected_fewshot = fewshot_examples_typeB
        conversation = fewshot_examples + [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text_input + "\n<think>\n"}]
            #"content": content
        }]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)


        stopping_criteria = StoppingCriteriaList([StopAfterThinkAnswer(processor)])

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        gen_len = generated_ids.shape[0]
        print(f"Generated token count: {gen_len}")

        decoded = processor.decode(generated_ids, skip_special_tokens=True)

    # extract the reasoning and answer
    pred, is_correct = extract_pred_answer(decoded, answer)

    result_entry = {
        "question": question,
        "answer": answer,
        "image_path": image_path,
        "subject": subject,
        "choices": choices,
        f"{model_name}_result": decoded,
        "predicted_answer": pred,
        "correct": is_correct,
        "has_think_tag": "</think>" in decoded.lower(),
        "truncated": gen_len >= 1024
    }


    if result_entry["correct"]:
        correct_count += 1

    results.append(result_entry)


#save output file
output_file = f"results_{model_name}_{dataset_name}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[INFO] Results saved to {output_file}")