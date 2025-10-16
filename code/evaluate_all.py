import json
import os

base_path = "/home/yaga9887/"

models = [
    "llava",
    "linear_0.25_final_notk",
    "linear_0.5_final_notk",
    "linear_0.75_final_notk",
    "linear_1.0_final_notk"
]

datasets = [
    "realworldqa",
    "ai2d",
    "mme",
    "mmmu",
    "mathvista"
    # "mmlu"
]

def evaluate(pred_file_path, correct_key="correct"):
    with open(pred_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(pred_file_path)
    total = len(data)
    correct = sum(1 for ex in data if ex.get(correct_key, False))
    think_samples = [ex for ex in data if ex.get("has_think_tag", False)]
    think_count = len(think_samples)
    think_correct = sum(1 for ex in think_samples if ex.get(correct_key, False))
    not_truncated_count = sum(1 for ex in data if not ex.get("truncated", False))

    # print(f"\nEvaluating: {os.path.basename(pred_file_path)}")
    # print(f"Total Samples: {total}")
    #print(f"Overall Accuracy: {correct / total:.2%}")
    #print(f"Samples with </think>: {think_count} ({think_count / total:.2%})")
    #print(f"Accuracy on samples with </think>: {think_correct / total:.2%}" if think_count > 0 else "No samples with </think>")
    #print(f"Samples with generation length < 1024 tokens: {not_truncated_count} ({not_truncated_count / total:.2%})")

def main():
    for model in models:
        for dataset in datasets:
            file_name = f"results_{model}_{dataset}.json"
            file_path = os.path.join(base_path, file_name)
            if os.path.exists(file_path):
                evaluate(file_path)
            else:
                print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
