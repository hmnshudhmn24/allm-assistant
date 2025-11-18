import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.dataset import load_jsonl, preprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_prompts(example):
    # concatenates prompt and response for causal LM training
    return {'text': example['prompt'] + '\n' + example.get('response','') + '\n'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='gpt2')
    parser.add_argument('--train_file', default='data/train.jsonl')
    parser.add_argument('--validation_file', default='data/val.jsonl')
    parser.add_argument('--output_dir', default='outputs/allm')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    train_ds = load_jsonl(args.train_file).map(preprocess)
    val_ds = load_jsonl(args.validation_file).map(preprocess)

    train_text = train_ds.map(make_prompts, remove_columns=train_ds.column_names)
    val_text = val_ds.map(make_prompts, remove_columns=val_ds.column_names)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)

    train_tok = train_text.map(tokenize, batched=True, remove_columns=train_text.column_names)
    val_tok = val_text.map(tokenize, batched=True, remove_columns=val_text.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        logging_steps=50,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator
    )

    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()
