import os
import json
from PIL import Image
from datasets import load_dataset

# Where to save the toy dataset
save_dir = "/media/chagan/yaga9887/dataset/LLaVA-Bench-Wilder"
image_dir = os.path.join(save_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# Load the dataset
dataset = load_dataset("lmms-lab/LLaVA-Bench-Wilder")["test"]
# subset = dataset.select(range(20)) 

data_list = []

for i, example in enumerate(dataset):
    try:
        image = example["image"]  
        image_path = os.path.join(image_dir, f"image_{i}.png")
        image.save(image_path)

        metadata = {
            "question": example.get("question", ""),
            "image_path": image_path,
            "answer": example.get("answer", "")
        }

        if example.get("choices") not in [None, "none"]:
            metadata["choices"] = example["choices"]

        data_list.append(metadata)

    except Exception as e:
        print(f"Failed on example {i}: {e}")
        continue

# Save to JSON
json_save_path = os.path.join(save_dir, "LLaVA-Bench Wilder.json")
with open(json_save_path, "w") as f:
    json.dump(data_list, f, indent=2)

print(f"Saved {len(data_list)} examples to: {json_save_path}")
print(f"Images saved to: {image_dir}")





#MMMU
import os
import json
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
import random

# Where to save the toy dataset
#save_dir = "/media/chagan/yaga9887/dataset/mmmu"
save_dir = "/home/yaga9887/toy_data/mmmu"
#os.makedirs(save_dir, exist_ok=True)
image_dir = os.path.join(save_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# Load the dataset
dev_data = load_dataset("lmms-lab/MMMU", split="dev")
val_data = load_dataset("lmms-lab/MMMU", split="validation")
combined = concatenate_datasets([dev_data, val_data])
filtered = combined.filter(lambda x: x["question_type"] == "multiple-choice")
subset = filtered.select(range(min(1000, len(filtered))))

#grouped_by_cat = defaultdict(list)
#for item in dataset:
#    grouped_by_cat[item["category"]].append(item)
#N = 1000
#subset = dataset.select(range(14, 24))
#total_count = sum(len(v) for v in grouped_by_cat.values())

#for cat, items in grouped_by_cat.items():
#    cat_n = int(len(items) / total_count * N)
#    subset.extend(random.sample(items, cat_n))
print(f"length:{len(subset)}")

data_list = []

for i, example in enumerate(subset):
    try:
        image_keys = [k for k in example.keys() if k.startswith("image_")]
        image_paths = []
        for j, key in enumerate(sorted(image_keys)):  
            image = example[key]
            if image is None:
                continue
            image_path = os.path.join(image_dir, f"image_{i}_{j+1}.png")
            image.save(image_path)
            image_paths.append(image_path)

        if "explanation" in example and example['explanation']:
            question_text = f"{example['question']}\nExplanation: {example['explanation']}"
        else:
            question_text = example['question']

        metadata = {
            "question": question_text,
            "image_path": image_paths,
            "answer": example.get("answer", "")
        }

        if example.get("options") not in [None, "none"]:
            metadata["choices"] = example["options"]

        data_list.append(metadata)

    except Exception as e:
        print(f"Failed on example {i}: {e}")
        continue

# Save to JSON
json_save_path = os.path.join(save_dir,"mmmu_toy.json")
with open(json_save_path, "w") as f:
    json.dump(data_list, f, indent=2)

print(f"Saved {len(data_list)} examples to: {json_save_path}")
#print(f"Images saved to: {image_dir}")




#mathvista
import os
import json
from PIL import Image
from datasets import load_dataset

# Where to save the toy dataset
#save_dir = "/media/chagan/yaga9887/dataset/LLaVA-Bench-Wilder"
save_dir = "/home/yaga9887/toy_data/mathvista"
image_dir = os.path.join(save_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# Load the dataset
dataset = load_dataset("AI4Math/MathVista")["testmini"]
subset = dataset.select(range(20)) 

data_list = []

for i, example in enumerate(subset):
    try:
        question_type = example.get("question_type")
        image = example["decoded_image"]  
        image_path = os.path.join(image_dir, f"image_{i}.png")
        image.save(image_path)

        metadata = {
            "question": example["question"],
            "question_type": question_type,
            "image_path": image_path,
            "answer": example["answer"]
        }

        if question_type == "multi_choice":
            metadata["choices"] = example["choices"]

        data_list.append(metadata)

    except Exception as e:
        print(f"Failed on example {i}: {e}")
        continue

# Save to JSON
json_save_path = os.path.join(save_dir, "mathvista_toy.json")
with open(json_save_path, "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data_list)} examples to: {json_save_path}")


#mmlu
import os
import json
import random
from datasets import load_dataset
from collections import Counter, defaultdict

# Directory setup
save_dir = "/media/chagan/yaga9887/dataset/mmlu"
image_dir = os.path.join(save_dir, "images")
os.makedirs(image_dir, exist_ok=True)

N = 1000  # Total number of samples to draw

# Load the 'all' config with the 'test' split
dataset = load_dataset("cais/mmlu", "all", split="test")

# Count how many samples per subject
subject_counts = Counter(dataset["subject"])

# Calculate sampling proportions
total = sum(subject_counts.values())
subject_proportions = {subj: count / total for subj, count in subject_counts.items()}

print(f"Subjects and their proportions in dataset:")
for subj, prop in subject_proportions.items():
    print(f"  {subj}: {prop:.4f}")

# Prepare data grouped by subject
data_by_subject = defaultdict(list)
for example in dataset:
    data_by_subject[example["subject"]].append(example)

# Sample from each subject proportionally
sampled = []
for subj, examples in data_by_subject.items():
    n_samples = int(subject_proportions[subj] * N)
    n_samples = min(n_samples, len(examples))  # avoid oversampling
    sampled.extend(random.sample(examples, n_samples))

print(f"Total sampled examples: {len(sampled)}")

# Save metadata for JSON (no images)
data_list = []
for i, example in enumerate(sampled):
    try:
        metadata = {
            "question": example.get("question", ""),
            "answer": example.get("answer", ""),
            "subject": example.get("subject"),
        }

        # Add choices if present and valid
        choices = example.get("choices")
        if choices not in [None, "none"]:
            metadata["choices"] = choices

        data_list.append(metadata)
    except Exception as e:
        print(f"Failed to process example {i}: {e}")
        continue

# Save metadata JSON
json_save_path = os.path.join(save_dir, "mmlu_toy.json")
with open(json_save_path, "w") as f:
    json.dump(data_list, f, indent=2)

print(f"Saved {len(data_list)} examples to {json_save_path}")
