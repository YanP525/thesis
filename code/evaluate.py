import json
import argparse
import os
import sys

base_path = "/home/yaga9887/"

def evaluate(pred_file_path, correct_key="correct"):
    # Load prediction data from JSON file
    with open(pred_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    # Count how many samples are correct
    correct = sum(1 for ex in data if ex.get(correct_key, False))
    # Count how many samples contain the </think> tag
    think_samples = [ex for ex in data if ex.get("has_think_tag", False)]
    think_count = len(think_samples)
    # Count how many samples with </think> are correct
    think_correct = sum(1 for ex in think_samples if ex.get(correct_key, False))
    # Count how many samples are not truncated (generation length < 1024 tokens)
    not_truncated_count = sum(1 for ex in data if not ex.get("truncated", False))

    # Collect all keys that end with '_result' (these contain generation texts)
    result_keys = set()
    for ex in data:
        for k in ex.keys():
            if k.endswith("_result"):
                result_keys.add(k)

    # Initialize accumulators for total length and counts per '_result' field
    length_stats = {k: 0 for k in result_keys}
    count_stats = {k: 0 for k in result_keys}

    # Accumulate lengths for each '_result' field
    for ex in data:
        for k in result_keys:
            if k in ex and isinstance(ex[k], str):
                length_stats[k] += len(ex[k])
                count_stats[k] += 1

    print(f"\nEvaluating: {os.path.basename(pred_file_path)}")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {correct / total:.2%}")
    print(f"Samples with </think>: {think_count / total:.2%}")
    print(f"Accuracy on samples with </think>: {think_correct / total:.2%}" if think_count > 0 else "No samples with </think>")
    print(f"Samples with generation length < 1024 tokens: {not_truncated_count / total:.2%}")

    # Print average length of each '_result' field
    for k in sorted(result_keys):
        if count_stats[k] > 0:
            avg_len = length_stats[k] / count_stats[k]
            print(f"Average length of '{k}': {avg_len:.2f} chars (based on {count_stats[k]} samples)")

    # Compute confusion matrix based on presence of </think> tag and correctness
    a = sum(1 for ex in data if ex.get("has_think_tag", False) and ex.get(correct_key, False))
    b = sum(1 for ex in data if ex.get("has_think_tag", False) and not ex.get(correct_key, False))
    c = sum(1 for ex in data if not ex.get("has_think_tag", False) and ex.get(correct_key, False))
    d = sum(1 for ex in data if not ex.get("has_think_tag", False) and not ex.get(correct_key, False))

    print("\nConfusion Matrix (</think> tag vs correctness):")
    print(f"{'':<20} {'Correct':<10} {'Incorrect':<10} {'Total':<10}")
    print(f"{'has </think>':<20} {a:<10} {b:<10} {a + b:<10}")
    print(f"{'no </think>':<20} {c:<10} {d:<10} {c + d:<10}")
    print(f"{'Total':<20} {a + c:<10} {b + d:<10} {total:<10}")


if __name__ == "__main__":
    pred_file = sys.argv[1]
    print(f"Loading prediction file: {pred_file}")

    evaluate(pred_file_path=base_path + pred_file)
