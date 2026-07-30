import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from src.faqs import load_faqs

MODEL = "gpt-5.4-mini"
INPUT_PRICE = 0.75 / 1_000_000
OUTPUT_PRICE = 4.50 / 1_000_000

SYSTEM_PROMPT = """You are a data augmentation assistant. Given a FAQ entry, generate 5 diverse natural-language query variants that a customer might type when searching for this information.

Rules:
- Use varied phrasing, synonyms, and sentence structures
- Queries should sound like real customer questions (including typos or informal language occasionally)
- Each query must be answerable by the given FAQ entry
- Return ONLY a JSON object with a "questions" array of 5 strings"""


class Questions(BaseModel):
    questions: list[str]


def _generate_variants(client, faq, index):
    text = f"Category: {faq['Category']}\nQuestion: {faq['Question']}\nAnswer: {faq['Answer']}"
    doc_id = f"faq_{index}"

    for attempt in range(4):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                text_format=Questions,
            )
            questions = response.output[0].content[0].parsed.questions
            return [(q, doc_id) for q in questions]
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed for FAQ {index} ({faq['Question']}): {e}")
    return []


def main():
    faqs = load_faqs()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        print("OPENAI_API_KEY not set")
        return

    total_input_tokens = 0
    total_output_tokens = 0
    all_rows = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_generate_variants, client, faq, i): i for i, faq in enumerate(faqs)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating queries"):
            result = future.result()
            all_rows.extend(result)

    rows_needed = len(faqs) * 5
    print(f"Generated {len(all_rows)} queries from {len(faqs)} FAQs")

    output_path = os.path.join(os.path.dirname(__file__), "data", "ground_truth.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "document_id"])
        for question, doc_id in all_rows:
            writer.writerow([question, doc_id])

    print(f"Written to {output_path}")
    print(f"Pricing: ${len(all_rows) * INPUT_PRICE:.4f} input, ${len(all_rows) * OUTPUT_PRICE:.4f} output")


if __name__ == "__main__":
    main()
