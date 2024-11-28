import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

def main():
    parser = argparse.ArgumentParser(description="Train, validate, or predict with a classifier.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--train_file", type=str, help="Path to the training JSONL file.")
    parser.add_argument("--validation_file", type=str, help="Path to the validation JSONL file.")
    parser.add_argument("--question_column", type=str, default="question", help="Column name for questions.")
    parser.add_argument("--answer_column", type=str, default="answer", help="Column name for answers.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training.")
    parser.add_argument("--max_seq_length", type=int, default=384, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--doc_stride", type=int, default=128, help="Stride size for splitting long sequences.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=100, help="Evaluation batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the outputs.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cache if set.")
    parser.add_argument("--do_train", action="store_true", help="Run training if set.")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation if set.")
    args = parser.parse_args()

    # Load dataset
    datasets = {}
    if args.train_file and args.do_train:
        datasets['train'] = load_dataset('json', data_files=args.train_file, split='train')
    if args.validation_file and args.do_eval:
        datasets['validation'] = load_dataset('json', data_files=args.validation_file, split='train')

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)

    def preprocess_function(examples):
        return tokenizer(
            examples[args.question_column],
            examples[args.answer_column],
            truncation=True,
            max_length=args.max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=False,
            padding="max_length",
        )

    # Tokenize dataset
    tokenized_datasets = {
        split: datasets[split].map(preprocess_function, batched=True) for split in datasets
    }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch" if args.do_eval else "no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        load_best_model_at_end=True if args.do_eval else False,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'] if args.do_train else None,
        eval_dataset=tokenized_datasets['validation'] if args.do_eval else None,
        tokenizer=tokenizer,
    )

    # Training
    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)

    # Evaluation
    if args.do_eval:
        metrics = trainer.evaluate()
        print(metrics)

if __name__ == "__main__":
    main()
