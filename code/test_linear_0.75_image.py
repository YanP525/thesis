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
fewshot_examples = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Why might someone go to this place? To go shopping?\nPlease answer yes or no.",
            "answer": "Yes"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Okay, so I'm trying to figure out why someone might go to this place, and the options are \"To go shopping?\" with the choices being yes or no. Hmm, first, I need to understand the context. The previous questions were about investments, project management, and scheduling exams, but now it's a different topic. \n\nAlright, so the question is asking if someone would go to this place specifically to go shopping. I'm not entirely sure what \"this place\" refers to because it wasn't mentioned before. Maybe it's a new location or a hypothetical place. Since the previous questions were more about math and logic, maybe \"this place\" is related to one of those topics indirectly.\n\nIf I think about investment places, maybe it's a financial institution like a bank or a stock exchange. Would someone go there to shop? Well, people might go to a bank to invest money or take out loans, but is that considered shopping? Not exactly, because shopping usually implies buying goods or services for personal use, not financial transactions.\n\nLooking at the project management question, it was about scheduling tasks. Maybe \"this place\" is a project management office or a workspace. Would someone go there to shop? Probably not, unless they're buying supplies, but that's more of a store visit, not necessarily going to an office.\n\nThe exam scheduling question involved students taking exams. If \"this place\" is a school or an exam center, would someone go there to shop? Students might buy textbooks or supplies, but again, that's more about shopping, but the primary purpose is education.\n\nWait, maybe \"this place\" is a store. If it's a store, then yes, people go there to shop. But the previous questions didn't mention a store. Maybe it's a trick question. If I have to answer based on the options given, I need to decide if going to this place is for shopping or not.\n\nConsidering all possibilities, without more context, it's hard to be certain. But since the previous questions were more about tasks and exams, maybe \"this place\" is a school or an office, and shopping isn't the primary reason. So, the answer might be no.\n\nBut I'm not entirely sure. Maybe I should lean towards yes because the question is specifically asking if someone might go there to shop, implying that it's possible. I'll go with yes.\n</think>\n\nYes"}
        ]
    },

    {
       "role": "user",
       "content": [
           {
           "type": "text",
           "text": "Is this book possibly written by a Franch writer?\nPlease answer yes or no.",
          "answer": "No"
           }
       ]
    },
    {
       "role": "assistant",
       "content": [
           {"type": "text", "text": "Okay, so I'm trying to figure out whether this book is possibly written by a French writer. The user has given me a series of questions and answers, and now they're asking about the book's authorship. I need to determine if the book is by a French writer, and the options are \"yes\" or \"no.\" \n\nFirst, I should look at the previous interactions to see if there's any context that might help. The initial questions were about investments, project management, and scheduling, which don't directly relate to literature or authors. So, I don't have any specific information about the book itself or its content from those previous exchanges.\n\nSince there's no direct information provided about the book, I need to make an assumption based on the user's possible intent. Maybe the book is mentioned in a way that hints at a French origin, like if the title is in French or if the content is related to French culture. But without that information, it's hard to be certain.\n\nI should consider that the user might be testing my ability to infer based on limited information. Perhaps the book's title or some clues in the text suggest a French author. For example, if the book is about French literature, history, or culture, that could be a hint. Alternatively, if the author's name sounds French, that would be a clear indicator.\n\nHowever, since I don't have access to the book's title or any specific details, I can't definitively say yes or no. I should inform the user that I don't have enough information to make that determination. It's important to be honest and not make an assumption that could be incorrect.\n\nIn summary, without additional context or specific information about the book, I can't accurately assess whether it's written by a French writer. I should respond that I don't have enough data to answer that question.\n</think>\n\nNo"}
       ]
    },
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Is this desk used for working?\nPlease answer yes or no.",
            "answer": "Yes"
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text":  "Okay, so I'm trying to figure out whether the desk is used for working. First, I need to understand what the question is asking. It's a yes or no question, so I just need to determine if the desk has a function related to work.\n\nI know that a desk is a piece of furniture where people usually work, like writing, studying, or using a computer. So, it's likely that the desk is used for working. I don't see any information that would suggest otherwise. Therefore, the answer should be yes.\n</think>\n\nYes"}
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
    #indexed_choices = [f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)]
    choice_str = "\n".join(indexed_choices)

    prompt = f"{question}\n{choice_str}\nPlease answer directly with only the number of the correct option and nothing else."
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
    #subject = item['subject']

    #construct different prompt accoring to question types
    # if choices:
    #    #choices_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
    #    text_input = f"Q: {question}\n{choices}\nPlease select one choice and write out the full answer and nothing else."
    # else:
    text_input = f"Q: {question}\nPlease answer yes or no."

    #text_input = build_prompt(question, choices)

    if model_name.startswith("deepseek"):
        fewshot_examples = convert_to_deepseek_format(fewshot_examples)
        conversation = fewshot_examples + [{"role": "user", "content": text_input + "\n<think>\n"}]
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
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text_input + "\n<think>\n"}]
            #"content": [{"type": "image"}, {"type": "text", "text": question + "\n<think>\n"}]
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
        #"subject": subject,
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