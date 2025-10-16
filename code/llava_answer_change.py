import re
import json
import shutil

def update_predictions_from_llava_result(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0

    for entry in data:
        decoded = entry.get("llava_result", "")  
        gt_answer = entry.get("answer", "")

        match = re.search( #\nThe correct answer is:(\n) 0. Gregory
            r'is[:\s]*\n?\s*([A-D0-9])(?:[.)\s]|$)',
            decoded,
            re.IGNORECASE
        )

        if match:
            pred_raw = match.group(1)
        else:
            lines = [line.strip() for line in decoded.strip().splitlines() if line.strip()]
            if lines:
                pred_raw = lines[0][0]
            else:
                pred_raw = ""

        pred_clean = str(pred_raw).lower().strip().rstrip(".")
        gt_clean = str(gt_answer).lower().strip().rstrip(".")
        is_correct = pred_clean == gt_clean

        entry["predicted_answer"] = pred_raw
        entry["correct"] = is_correct
        updated_count += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{updated_count} updates")


if __name__ == "__main__":
    update_predictions_from_llava_result("results_llava_mmlu.json")
