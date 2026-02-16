"""
Benchmark medical LLMs on MMLU medical subsets, broken down by specialty.

Measures accuracy per model per medical subject to quantify how performance
varies across clinical contexts. Uses GitHub Models API for inference.
"""

import csv
import json
import os
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

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-5",
    "meta/llama-4-scout-17b-16e-instruct",
    "deepseek/deepseek-r1",
    "mistral-ai/mistral-medium-2505",
]

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

    # Extract answer letter from response
    parsed = None
    for char in raw.upper():
        if char in "ABCD":
            parsed = char
            break
    return parsed, raw


def run_benchmark():
    results = []
    responses = []

    for subject in MEDICAL_SUBJECTS:
        print(f"\n--- {subject} ---")
        dataset = load_dataset("cais/mmlu", subject, split="test")

        for model_id in MODELS:
            correct = 0
            total = 0
            errors = 0

            for qi, row in enumerate(dataset):
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
                        print(f"  Rate limited on {model_id}, waiting 60s...")
                        time.sleep(60)
                        try:
                            prediction, raw = query_model(model_id, question, choices)
                            if prediction == answer:
                                correct += 1
                            total += 1
                        except Exception:
                            errors += 1
                    else:
                        errors += 1
                except Exception:
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

                time.sleep(0.2)

            accuracy = correct / total if total > 0 else 0
            print(f"  {model_id}: {accuracy:.1%} ({correct}/{total}, {errors} errors)")

            results.append({
                "subject": subject,
                "model": model_id,
                "correct": correct,
                "total": total,
                "errors": errors,
                "accuracy": round(accuracy, 4),
            })

    return results, responses


def write_results(results, responses):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate accuracy per model per subject
    csv_path = OUTPUT_DIR / "mmlu_by_specialty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "model", "correct", "total", "errors", "accuracy"])
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "mmlu_by_specialty.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Per-question raw responses for auditability
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
