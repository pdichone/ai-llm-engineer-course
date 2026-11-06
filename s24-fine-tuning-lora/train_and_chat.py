import os
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import platform
from pathlib import Path
import time
import psutil
from prettytable import PrettyTable


# Universal device setup
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA (GPU)")
        return torch.device("cuda")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        # For Mac M1/M2
        print("Mac M1/M2 detected. Using CPU for better compatibility")
        return torch.device("cpu")
    else:
        print("Using CPU")
        return torch.device("cpu")


# Universal path handling
def get_save_path(folder_name):
    return str(Path(os.getcwd()) / folder_name)


class ModelMetrics:
    def __init__(self):
        self.metrics = {
            "original_size": 0,
            "lora_size": 0,
            "training_time": 0,
            "memory_usage": 0,
            "parameter_count": 0,
            "trainable_params": 0,
            "compression_ratio": 0,
            "accuracy": 0,
        }


class ModelTrainer:
    def __init__(self):
        self.device = get_device()
        self.model_checkpoint = "distilbert-base-uncased"
        self.output_dir = get_save_path("checkpoints")
        self.final_model_dir = get_save_path("lora-sentiment-improved")
        self.metrics = ModelMetrics()

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.final_model_dir, exist_ok=True)

    def prepare_movie_dataset(self):
        print("Loading IMDB dataset...")
        imdb_dataset = load_dataset("imdb", split="train[:2000]")

        def convert_labels(example):
            return {
                "text": example["text"],
                "labels": int(example["label"]),  # Ensure labels are integers
            }

        print("Processing IMDB dataset...")
        imdb_dataset = imdb_dataset.map(convert_labels)

        print("Adding custom movie examples...")
        additional_examples = {
            "text": [
                "I absolutely loved the new Spider-Man movie! The special effects were amazing!",
                "The movie was fantastic! Great special effects and acting!",
                "Amazing visual effects and compelling story!",
                "This film was a masterpiece! Incredible performances!",
                "The special effects were mind-blowing! Loved every minute!",
                "The worst movie ever. Terrible acting and effects.",
                "Such a disappointing film. Waste of time.",
                "The movie was boring and predictable.",
            ],
            "labels": [1, 1, 1, 1, 1, 0, 0, 0],  # Make sure these are integers
        }

        additional_dataset = Dataset.from_dict(additional_examples)

        columns_to_keep = ["text", "labels"]
        imdb_dataset = imdb_dataset.remove_columns(
            [col for col in imdb_dataset.column_names if col not in columns_to_keep]
        )

        print("Combining datasets...")
        combined_dataset = concatenate_datasets([imdb_dataset, additional_dataset])
        print(f"Final dataset size: {len(combined_dataset)} examples")
        return combined_dataset

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
            f"|| trainable%: {100 * trainable_params / all_param:.2f}"
        )
        return trainable_params, all_param

    def build_lora_model(self, num_labels):
        print("Building LoRA model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=num_labels
        )

        # Get original model size
        self.metrics.metrics["original_size"] = sum(
            p.numel() for p in model.parameters()
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )

        lora_model = get_peft_model(model, lora_config)

        # Get LoRA model metrics
        trainable, total = self.print_trainable_parameters(lora_model)
        self.metrics.metrics["trainable_params"] = trainable
        self.metrics.metrics["parameter_count"] = total
        self.metrics.metrics["compression_ratio"] = total / trainable

        return lora_model

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return evaluate.load("accuracy").compute(
            predictions=predictions, references=labels
        )

    def train(self):
        try:
            print(f"Starting training process on {self.device}...")
            start_time = time.time()
            metric = evaluate.load("accuracy")

            training_args = TrainingArguments(
                output_dir=self.output_dir,
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                logging_steps=10,
                warmup_steps=500,
                seed=42,
                use_cpu=self.device.type == "cpu",  # Updated from no_cuda
            )

            # Prepare dataset
            dataset = self.prepare_movie_dataset()
            train_test_split = dataset.train_test_split(test_size=0.2)

            # Initialize tokenizer
            print("Initializing tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

            def preprocess_data(examples):
                # Tokenize the texts
                tokenized = tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                )
                # Add labels
                tokenized["labels"] = examples["labels"]
                return tokenized

            # Process datasets
            print("Processing training dataset...")
            train_dataset = train_test_split["train"].map(
                preprocess_data,
                batched=True,
                remove_columns=train_test_split["train"].column_names,
            )

            print("Processing test dataset...")
            test_dataset = train_test_split["test"].map(
                preprocess_data,
                batched=True,
                remove_columns=train_test_split["test"].column_names,
            )

            # Set format
            columns = ["input_ids", "attention_mask", "labels"]
            train_dataset.set_format(type="torch", columns=columns)
            test_dataset.set_format(type="torch", columns=columns)

            # Create data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer, padding=True, return_tensors="pt"
            )

            # Build model
            model = self.build_lora_model(num_labels=2)
            model.to(self.device)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )

            print("\nStarting training...")
            trainer.train()

            # Update metrics
            self.metrics.metrics["training_time"] = time.time() - start_time
            self.metrics.metrics["memory_usage"] = (
                psutil.Process().memory_info().rss / 1024 / 1024
            )

            print("\nSaving model...")
            trainer.save_model(self.final_model_dir)
            tokenizer.save_pretrained(self.final_model_dir)

            # Evaluate final model
            eval_results = trainer.evaluate()
            self.metrics.metrics["accuracy"] = eval_results["eval_accuracy"]

            self.print_training_summary()

            return model, tokenizer

        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            raise

    def print_training_summary(self):
        summary = PrettyTable()
        summary.field_names = ["Metric", "Value"]
        summary.align["Metric"] = "l"
        summary.align["Value"] = "r"

        metrics = self.metrics.metrics
        summary.add_row(["Training Time", f"{metrics['training_time']:.2f} seconds"])
        summary.add_row(["Memory Usage", f"{metrics['memory_usage']:.2f} MB"])
        summary.add_row(["Total Parameters", f"{metrics['parameter_count']:,}"])
        summary.add_row(["Trainable Parameters", f"{metrics['trainable_params']:,}"])
        summary.add_row(["Compression Ratio", f"{metrics['compression_ratio']:.2f}"])
        summary.add_row(["Final Accuracy", f"{metrics['accuracy']:.2%}"])

        print("\nTraining Summary:")
        print(summary)


class ModelInterface:
    def __init__(self):
        self.device = get_device()
        self.tokenizer = None
        self.sentiment_model = None
        self.model_checkpoint = "distilbert-base-uncased"

    def load_models(self):
        try:
            print("Loading models and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

            if os.path.exists("./lora-sentiment-improved"):
                base_model_sentiment = (
                    AutoModelForSequenceClassification.from_pretrained(
                        self.model_checkpoint, num_labels=2
                    )
                )
                self.sentiment_model = PeftModel.from_pretrained(
                    base_model_sentiment, "./lora-sentiment-improved"
                )
                self.sentiment_model.eval()
                self.sentiment_model.to(self.device)
            else:
                raise FileNotFoundError(
                    "Sentiment model not found. Please train the model first."
                )

            print("Models loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def analyze_text(self, text):
        if not text.strip():
            return None

        inputs = self.tokenizer(
            text.lower().strip(), return_tensors="pt", truncation=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()

                return {
                    "sentiment": {
                        "prediction": "Positive" if prediction == 1 else "Negative",
                        "confidence": confidence,
                    }
                }

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return None


def main():
    print("\n" + "=" * 50)
    print("Movie Review Sentiment Analysis System")
    print("=" * 50)

    print(f"\nSystem Information:")
    print(f"Operating System: {platform.system()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if platform.system() == "Darwin":
        print(f"MPS available: {torch.backends.mps.is_available()}")
    print("=" * 50 + "\n")

    while True:
        print("\nChoose an option:")
        print("1. Train new model")
        print("2. Analyze reviews using existing model")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            print("\nInitiating model training...")
            trainer = ModelTrainer()
            try:
                trainer.train()
                print("\nTraining completed successfully!")
            except Exception as e:
                print(f"\nTraining failed: {str(e)}")

        elif choice == "2":
            interface = ModelInterface()
            if interface.load_models():
                print("\nModel loaded successfully!")
                while True:
                    print(
                        "\nEnter a movie review to analyze (or 'quit' to go back to main menu):"
                    )
                    text = input().strip()

                    if text.lower() == "quit":
                        break

                    if text:
                        print("\nAnalyzing...")
                        result = interface.analyze_text(text)
                        if result:
                            sentiment = result["sentiment"]
                            print(f"\nSentiment: {sentiment['prediction']}")
                            print(f"Confidence: {sentiment['confidence']:.2%}")
                        else:
                            print("Analysis failed. Please try again.")
            else:
                print("\nFailed to load model. Please train a model first.")

        elif choice == "3":
            print("\nThank you for using the Movie Review Sentiment Analysis System!")
            break

        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
    finally:
        print("\nExiting program...")
