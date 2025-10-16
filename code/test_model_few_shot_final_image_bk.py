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
#data_base_path = "/home/yaga9887/toy_data"
data_base_path = "/media/chagan/yaga9887/dataset"

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
fewshot_examples_typeA = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You invest $1 at a 5% interest rate for 3 years. What is the present value of that $1?\nChoices: [\"0.8638\", \"1\", \"14\", \"10\"]\nPlease select one choice and write out the full answer.",
            "answer": "0.8638"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "To determine the present value of \\$1 invested at a 5% annual interest rate for 3 years, I'll use the present value formula for a compound interest.\n\nFirst, I'll identify the variables: the principal amount (P) is \\$1, the annual interest rate (r) is 5%, and the number of years (n) is 3.\n\nNext, I'll convert the interest rate to a decimal by dividing 5% by 100, which gives 0.05.\n\nThen, I'll apply the present value formula: PV = P × (1 + r)^n.\n\nPlugging in the values: PV = 1 × (1 + 0.05)^3.\n\nCalculating (1.05)^3 gives approximately 1.157625.\n\nFinally, multiplying by the principal amount: PV ≈ 1 × 1.157625 = 1.157625.\n\nSince the options provided are 0.8638, 1, 14, and 10, the closest match to the calculated present value is 0.8638.\n</think>\n\nTo determine the **present value** of \\$1 invested at a **5% annual interest rate** for **3 years**, we'll use the **present value formula** for compound interest:\n\n\\[\nPV = P \\times (1 + r)^n\n\\]\n\nWhere:\n- \\( PV \\) is the present value (what we want to find)\n- \\( P \\) is the principal amount (\\$1)\n- \\( r \\) is the annual interest rate (5% or 0.05)\n- \\( n \\) is the number of years (3)\n\n**Step 1:** Plug in the known values into the formula.\n\n\\[\nPV = 1 \\times (1 + 0.05)^3\n\\]\n\n**Step 2:** Calculate \\( (1 + 0.05)^3 \\).\n\n\\[\n(1.05)^3 = 1.05 \\times 1.05 \\times 1.05 = 1.157625\n\\]\n\n**Step 3:** Multiply by the principal amount.\n\n\\[\nPV = 1 \\times 1.157625 = 1.157625\n\\]\n\n**Step 4:** Compare the calculated present value with the given choices.\n\nThe calculated present value is approximately **\\$1.1576**, but the closest option provided is **\\$0.8638**.\n</think>\n\n0.8638"}
        ]
    },

    {
       "role": "user",
       "content": [
           {
           "type": "text",
           "text": "A small project includes several tasks with different durations. The longest path through the tasks takes 26 days. What is the total project completion time?\nChoices: [\"23\", \"25\", \"26\", \"30\"]\nPlease select one choice and write out the full answer.",
          "answer": "26"
           }
       ]
    },
    {
       "role": "assistant",
       "content": [
           {"type": "text", "text": "To determine the total project completion time, I need to understand the nature of the problem. It mentions a small project with several tasks that have different durations, and the longest path through these tasks takes 26 days. The question asks for the total project completion time, with the options being 23, 25, 26, or 30 days.\n\nFirst, I need to clarify what is meant by the \"longest path.\" In project management, the longest path typically refers to the critical path method (CPM) or critical chain, which is the sequence of tasks that determines the minimum time required to complete the project. This path is critical because it dictates the project's completion time.\n\nAssuming the project is a directed acyclic graph (DAG), each task has a specific duration, and the longest path represents the maximum time required to complete all tasks without any overlaps. If the longest path is 26 days, it suggests that the project cannot be completed in less than 26 days without delaying some tasks.\n\nHowever, the question asks for the total project completion time. If the longest path is 26 days, it implies that the project cannot be completed any faster than 26 days without compressing the schedule, which isn't typically possible without altering task durations or dependencies.\n\nTherefore, the total project completion time should be at least 26 days. Among the provided options, 26 is the most appropriate choice.\n</think>\n\nTo determine the total project completion time, let's analyze the given information:\n\n- **Longest Path Duration:** The problem states that the longest path through the tasks takes **26 days**. This implies that this path is the critical path, meaning it determines the minimum time required to complete the project.\n\n- **Critical Path Method (CPM):** In project management, the critical path is the longest sequence of tasks that cannot be delayed without pushing the project's end date. It dictates the total project duration.\n\n- **Total Project Completion Time:** Since the longest path (critical path) takes 26 days, the project cannot be completed in less than 26 days. Therefore, the total project completion time is **26 days**.\n</think>\n\n26"}
       ]
    },
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Seven students each take 3 different exams. No student can take more than one exam per day. What is the minimum number of days needed to schedule all exams?\nChoices: [\"1\", \"2\", \"3\", \"4\"]\nPlease select one choice and write out the full answer.",
            "answer": "4"
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text":  "To determine the minimum number of days needed to schedule all exams, I need to consider the constraints provided.\n\nEach student has three different exams, and no student can take more than one exam per day. This means that each student will need at least three separate days to complete all their exams.\n\nSince there are seven students, and each day can accommodate all seven students taking one exam each, the total number of exams is 7 students multiplied by 3 exams, which equals 21 exams.\n\nTo find the minimum number of days required, I divide the total number of exams by the number of exams that can be held each day. So, 21 exams divided by 7 exams per day equals 3 days.\n\nTherefore, the minimum number of days needed is 3.\n</think>\n\nTo determine the minimum number of days required to schedule all exams, let's break down the problem step by step.\n\n**Given:**\n- **7 students** each need to take **3 different exams**.\n- **No student** can take more than **1 exam per day**.\n\n**Step 1: Calculate the Total Number of Exams**\nEach student has 3 exams to take, so the total number of exams is:\n\\[ 7 \\text{ students} \\times 3 \\text{ exams/student} = 21 \\text{ exams} \\]\n\n**Step 2: Determine the Maximum Exams per Day**\nSince each day, each student can take **1 exam**, the maximum number of exams that can be held in a single day is equal to the number of students:\n\\[ 7 \\text{ exams/day} \\]\n\n**Step 3: Calculate the Minimum Number of Days**\nTo find the minimum number of days needed to schedule all 21 exams, divide the total number of exams by the maximum exams per day:\n\\[ \\frac{21 \\text{ exams}}{7 \\text{ exams/day}} = 3 \\text{ days} \\]\n</think>\n\n3"}
        ]
    }
]

fewshot_examples_typeB = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You invest $1 at a 5% interest rate for 3 years. What is the present value of that $1?\nPlease answer the question providing the final value.",
            "answer": "0.8638"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "To determine the present value of \\$1 invested at a 5% annual interest rate for 3 years, I'll use the present value formula for compound interest.\n\nFirst, I'll identify the principal amount, which is \\$1, the annual interest rate, which is 5%, and the number of years, which is 3.\n\nNext, I'll convert the annual interest rate to a decimal by dividing by 100, resulting in 0.05.\n\nThen, I'll apply the present value formula: PV = Principal × (1 + r)^n.\n\nPlugging in the values, I'll calculate (1 + 0.05)^3, which equals 1.157625.\n\nFinally, I'll multiply the principal amount by this factor: 1 × 1.157625, which gives the present value of \\$1.157625.\n</think>\n\nTo determine the **present value** of \\$1 invested at a 5% annual interest rate for 3 years, we can use the **present value formula** for compound interest:\n\n\\[\nPV = P \\times (1 + r)^n\n\\]\n\nWhere:\n- \\( PV \\) is the present value (the amount you want to find),\n- \\( P \\) is the principal amount (\\$1),\n- \\( r \\) is the annual interest rate (5% or 0.05),\n- \\( n \\) is the number of years (3).\n\n**Step-by-Step Calculation:**\n\n1. **Plug in the known values:**\n   \\[\n   PV = 1 \\times (1 + 0.05)^3\n   \\]\n\n2. **Calculate \\( (1 + 0.05)^3 \\):**\n   \\[\n   (1 + 0.05)^3 = 1.05^3 = 1.157625\n   \\]\n\n3. **Multiply by the principal amount:**\n   \\[\n   PV = 1 \\times 1.157625 = 1.157625\n   \\]\n\n**Final Answer:**\n\nThe present value of \\$1 invested at a 5% annual interest rate for 3 years is:\n\n\\[\n\\boxed{\\$1.157625}\n\\\n</think>\n\n0.1.157625"}
        ]
    },

    {
       "role": "user",
       "content": [
           {
           "type": "text",
           "text":  "A small project includes several tasks with different durations. The longest path through the tasks takes 26 days. What is the total project completion time?\nPlease answer the question providing the final value.",
          "answer": "26"
           }
       ]
    },
    {
       "role": "assistant",
       "content": [
           {"type": "text", "text": "To determine the total project completion time, I need to understand the nature of the tasks and their dependencies. Since the problem mentions a \"longest path\" through the tasks, it implies that the tasks are sequential and depend on one another.\n\nIf all tasks are dependent on the previous one, the total time required is simply the sum of all individual task durations. Therefore, adding up the durations of all tasks will give the total project completion time.\n</think>\n\nTo determine the total project completion time, we need to consider the nature of the tasks and their dependencies. The problem mentions a \"longest path\" through the tasks, which implies that the tasks are sequential and depend on one another. \n\nIf all tasks are dependent on the previous one, the total time required is simply the sum of all individual task durations. Therefore, adding up the durations of all tasks will give the total project completion time.\n\n**Step-by-Step Solution:**\n\n1. **Identify the total number of tasks.**\n2. **List the duration of each task.**\n3. **Sum all the durations.**\n\n**Let’s assume the tasks and their durations are as follows:**\n\n- **Task 1:** 1 day\n- **Task 2:** 2 days\n- **Task 3:** 3 days\n- **Task 4:** 4 days\n- **Task 5:** 5 days\n\n**Total Project Completion Time = 1 + 2 + 3 + 4 + 5 = 15 days**\n</think>\n\n15"}
       ]
    },
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Seven students each take 3 different exams. No student can take more than one exam per day. What is the minimum number of days needed to schedule all exams?\\nPlease answer the question providing the final value.",
            "answer": "4"
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text":  "To determine the minimum number of days needed to schedule all exams, I need to consider the constraints and the number of exams each student has to take.\n\nEach of the seven students has three different exams, so there are a total of 7 multiplied by 3, which equals 21 exams.\n\nSince no student can take more than one exam per day, each day can accommodate at most seven exams, one for each student.\n\nTo find the minimum number of days required, I divide the total number of exams by the number of exams that can be held each day: 21 divided by 7 equals 3.\n\nTherefore, a minimum of 3 days is needed to schedule all exams.\n</think>\n\nTo determine the minimum number of days required to schedule all exams, let's break down the problem step by step.\n\n**Given:**\n- **Number of students:** 7\n- **Exams per student:** 3\n- **Constraint:** No student can take more than one exam per day.\n\n**Total Exams to Schedule:**\nEach of the 7 students has 3 exams, so the total number of exams is:\n\\[\n7 \\text{ students} \\times 3 \\text{ exams/student} = 21 \\text{ exams}\n\\]\n\n**Exams per Day:**\nSince each student can take only one exam per day, the maximum number of exams that can be held in a single day is equal to the number of students, which is 7.\n\n**Minimum Number of Days:**\nTo find the minimum number of days needed to schedule all 21 exams, we divide the total number of exams by the number of exams that can be held each day:\n\\[\n\\frac{21 \\text{ exams}}{7 \\text{ exams/day}} = 3 \\text{ days}\n\\]\n</think>\n\n3"}
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


# def build_prompt(question, choices):
#     # for datasets which take indices as ground truth
#     #indexed_choices = [f"{i}. {choice}" for i, choice in enumerate(choices)]
#     choices = ast.literal_eval(choices) #make sure "choices" is a list not a string
#     indexed_choices = [f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)]
#     choice_str = "\n".join(indexed_choices)

#     prompt = f"{question}\n{choice_str}\nPlease answer directly with only the letter of the correct option and nothing else.\n<think>\n"
#     return prompt


# read file
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []
correct_count = 0

for item in tqdm(data):
    question = item['question']
    question_type = item["question_type"]
    answer = item['answer']
    choices = item.get('choices', [])
    image_path = item['image_path']

    #construct different prompt accoring to question types
    if choices:
       #choices_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
       text_input = f"Q: {question}\n{choices}\nPlease select one choice and write out the full answer and nothing else."
    else:
       text_input = f"Q: {question}\nPlease answer the question providing the final value and nothing else."
    #text_input = build_prompt(question, choices)

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
        #image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")
        # content = [{"type": "image"} for _ in images]
        # content.append({"type": "text", "text": text_input + "\n<think>\n"})
        if question_type == "multi_choice":
            selected_fewshot = fewshot_examples_typeA
        else:
            selected_fewshot = fewshot_examples_typeB
        conversation = selected_fewshot + [{
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