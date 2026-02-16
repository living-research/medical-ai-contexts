"""
Benchmark medical LLMs on MMLU medical subsets, broken down by specialty.

Measures accuracy per model per medical subject to quantify how performance
varies across clinical contexts. Uses GitHub Models API for inference.

Rate limits (GitHub Models, Free/Pro tier):
  High-tier (GPT-4o, GPT-5, DeepSeek-R1): 10 req/min, 50 req/day
  Low-tier (Llama, Mistral): 15 req/min, 150 req/day

We sample questions per subject to stay within daily limits.
"""

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset

API_URL = "https://models.github.ai/inference/chat/completions"
API_TOKEN = os.environ.get("GH_MODELS_TOKEN")

if not API_TOKEN:
    print("GH_MODELS_TOKEN not set", file=sys.stderr)
    sys.exit(1)

# High-tier models: 50 req/day → 8 questions/subject (48 total)
# Low-tier models: 150 req/day → 25 questions/subject (150 total)
MODELS = {
    "openai/gpt-4o": {"sample": 8, "delay": 7},
    "openai/gpt-5": {"sample": 8, "delay": 7},
    "deepseek/deepseek-r1": {"sample": 8, "delay": 7},
    "meta/llama-4-scout-17b-16e-instruct": {"sample": 25, "delay": 5},
    "mistral-ai/mistral-medium-2505": {"sample": 25, "delay": 5},
}

MEDICAL_SUBJECTS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "professional_medicine",
    "college_medicine",
    "college_biology",
]

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
OUTPUT_DIR = Path("data/processed")
SEED = 42


def query_model(model_id, question, choices):
    """Send a multiple-choice medical question to a model and return its answer letter."""
    prompt = f"""{question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Reply with ONLY the letter of the correct answer (A, B, C, or D)."""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()

    parsed = None
    for char in raw.upper():
        if char in "ABCD":
            parsed = char
            break
    return parsed, raw


def run_benchmark():
    results = []
    responses = []
    rng = random.Random(SEED)

    for subject in MEDICAL_SUBJECTS:
        print(f"\n--- {subject} ---")
        dataset = load_dataset("cais/mmlu", subject, split="test")
        all_indices = list(range(len(dataset)))

        for model_id, config in MODELS.items():
            sample_size = min(config["sample"], len(dataset))
            sampled = sorted(rng.sample(all_indices, sample_size))
            delay = config["delay"]

            correct = 0
            total = 0
            errors = 0

            for qi in sampled:
                row = dataset[qi]
                question = row["question"]
                choices = row["choices"]
                answer = ANSWER_MAP[row["answer"]]

                prediction = None
                raw = ""
                try:
                    prediction, raw = query_model(model_id, question, choices)
                    if prediction == answer:
                        correct += 1
                    total += 1
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        print(f"  Rate limited on {model_id}, waiting 90s...")
                        time.sleep(90)
                        try:
                            prediction, raw = query_model(model_id, question, choices)
                            if prediction == answer:
                                correct += 1
                            total += 1
                        except Exception:
                            errors += 1
                    else:
                        print(f"  HTTP {e.response.status_code} on {model_id}: {e}")
                        errors += 1
                except Exception as e:
                    print(f"  Error on {model_id}: {e}")
                    errors += 1

                responses.append({
                    "subject": subject,
                    "model": model_id,
                    "question_index": qi,
                    "expected": answer,
                    "predicted": prediction or "",
                    "correct": prediction == answer,
                    "raw_response": raw,
                })

                time.sleep(delay)

            accuracy = correct / total if total > 0 else 0
            print(f"  {model_id}: {accuracy:.1%} ({correct}/{total}, {errors} errors)")

            results.append({
                "subject": subject,
                "model": model_id,
                "correct": correct,
                "total": total,
                "sample_size": sample_size,
                "errors": errors,
                "accuracy": round(accuracy, 4),
            })

    return results, responses


def write_results(results, responses):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "mmlu_by_specialty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject", "model", "correct", "total", "sample_size", "errors", "accuracy",
        ])
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "mmlu_by_specialty.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    responses_path = OUTPUT_DIR / "mmlu_responses.csv"
    with open(responses_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject", "model", "question_index", "expected", "predicted", "correct", "raw_response",
        ])
        writer.writeheader()
        writer.writerows(responses)

    print(f"\nResults written to {csv_path}, {json_path}, {responses_path}")


if __name__ == "__main__":
    results, responses = run_benchmark()
    write_results(results, responses)
