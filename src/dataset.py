import json
from datasets import Dataset

PROMPT_TEMPLATE = """Instruction: {instruction}
Input: {input}
Response:"""

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return Dataset.from_list(records)

def preprocess(example):
    prompt = PROMPT_TEMPLATE.format(instruction=example.get('instruction',''), input=example.get('input',''))
    example['prompt'] = prompt
    example['response'] = example.get('output','')
    return example
