import os
import torch
import numpy as np
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

model_checkpoint = "distilbert-base-uncased"

def print_model_size(path):
    size = 0
    for f in os.scandir(path):
        size += os.path.getsize(f)
    print(f"Model size: {(size / 1e6):.2} MB")


def print_trainable_parameters(model, label):
    parameters, trainable = 0, 0
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    print(
        f"{label} trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)"
    )


def build_lora_model(num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
    )
    print_trainable_parameters(model, label="Base model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

    lora_model = get_peft_model(model, lora_config)
    print_trainable_parameters(lora_model, label="LoRA")
    return lora_model


def preprocess_function(examples, tokenizer):
    # Process text
    texts = [str(text).lower().strip() for text in examples["text"]]

    # Tokenize
    result = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None,  # Changed this to return lists
    )

    # Add labels
    result["labels"] = examples["labels"]

    return result



if __name__ == "__main__":
    print("Starting LoRA fine-tuning demo...")

    # model_checkpoint = "distilbert-base-uncased"
    print(f"Using model: {model_checkpoint}")

    # Load datasets
    print("\nLoading datasets...")
    dataset1 = load_dataset("imdb", split="train[:1000]")
    dataset2 = load_dataset("ag_news", split="train[:1000]")

    print(f"Dataset 1 size: {len(dataset1)} examples")
    print(f"Dataset 2 size: {len(dataset2)} examples")

    # Prepare datasets
    dataset1 = dataset1.rename_column("label", "labels")
    dataset2 = dataset2.rename_column("label", "labels")

    # Split datasets
    train_size = int(0.8 * len(dataset1))
    dataset1_train = dataset1.select(range(train_size))
    dataset1_test = dataset1.select(range(train_size, len(dataset1)))
    dataset2_train = dataset2.select(range(train_size))
    dataset2_test = dataset2.select(range(train_size, len(dataset2)))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    config = {
        "sentiment": {
            "train_data": dataset1_train,
            "test_data": dataset1_test,
            "num_labels": 2,
            "epochs": 5,
            "path": "./lora-sentiment",
        },
        "topic": {
            "train_data": dataset2_train,
            "test_data": dataset2_test,
            "num_labels": 4,
            "epochs": 5,
            "path": "./lora-topic",
        },
    }

    # Preprocess datasets
    print("Preprocessing datasets...")
    for cfg in config.values():
        cfg["train_data"] = cfg["train_data"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=["text"],  # Only remove text column
        )
        cfg["test_data"] = cfg["test_data"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=["text"],  # Only remove text column
        )

        # Set format for pytorch
        cfg["train_data"].set_format("torch")
        cfg["test_data"].set_format("torch")

    training_arguments = TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        warmup_steps=100,
        seed=42,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    for name, cfg in config.items():
        print(f"\nTraining {name} classifier...")

        model = build_lora_model(cfg["num_labels"])

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=cfg["train_data"],
            eval_dataset=cfg["test_data"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        eval_results = trainer.evaluate()
        print(f"Evaluation accuracy: {eval_results['eval_accuracy']:.4f}")

        trainer.save_model(cfg["path"])
        print_model_size(cfg["path"])

    # Prediction function
    def predict_text(text, model_path, num_labels, task_type):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        inputs = tokenizer(
            text.lower().strip(), return_tensors="pt", truncation=True, max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        if task_type == "sentiment":
            label_map = {0: "Negative", 1: "Positive"}
        else:
            label_map = {
                0: "World",
                1: "Sports",
                2: "Business",
                3: "Science/Technology",
            }

        return label_map[predicted_class], confidence

    # Test examples
    test_texts = [
        {
            "text": "This movie was absolutely fantastic! The acting was superb.",
            "model": "sentiment",
            "num_labels": 2,
            "task_type": "sentiment",
            "expected": "Positive",
        },
        {
            "text": "The worst film I've ever seen. Complete waste of time.",
            "model": "sentiment",
            "num_labels": 2,
            "task_type": "sentiment",
            "expected": "Negative",
        },
        {
            "text": "Tesla stock surges 20% after strong quarterly earnings report.",
            "model": "topic",
            "num_labels": 4,
            "task_type": "topic",
            "expected": "Business",
        },
        {
            "text": "New AI model achieves breakthrough in protein folding.",
            "model": "topic",
            "num_labels": 4,
            "task_type": "topic",
            "expected": "Science/Technology",
        },
    ]

    print("\nRunning predictions on test examples:")
    for test in test_texts:
        prediction, confidence = predict_text(
            test["text"],
            config[test["model"]]["path"],
            test["num_labels"],
            test["task_type"],
        )
        print(f"\nText: {test['text']}")
        print(f"Expected: {test['expected']}")
        print(f"Predicted: {prediction}")
        print(f"Confidence: {confidence:.2%}")