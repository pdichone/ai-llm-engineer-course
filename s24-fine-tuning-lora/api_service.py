from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import uvicorn
from typing import List, Dict
import numpy as np
from pathlib import Path
import json

## Must pip install: pip install fastapi uvicorn



class ReviewRequest(BaseModel):
    text: str


class BatchReviewRequest(BaseModel):
    texts: List[str]


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processing_time: float


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time: float
    
    
class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_checkpoint = "distilbert-base-uncased"
        # self.model_path = "./lora-sentiment-improved"
        self.model_path = "./lora-sentiment"
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the tokenizer and model."""
        try:
            print(f"Loading model on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint, num_labels=2
            )

            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    async def predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment for a single text."""
        try:
            import time

            start_time = time.time()

            # Tokenize
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128, padding=True
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()

            # Map prediction to sentiment
            sentiment = "Positive" if prediction == 1 else "Negative"

            processing_time = time.time() - start_time

            return {
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "processing_time": processing_time,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def predict_batch_sentiment(self, texts: List[str]) -> Dict:
        """Predict sentiment for a batch of texts."""
        try:
            import time

            start_time = time.time()

            results = []
            for text in texts:
                result = await self.predict_sentiment(text)
                results.append(result)

            total_time = time.time() - start_time

            return {"results": results, "total_processing_time": total_time}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        


# Initialize FastAPI app
app = FastAPI(
    title="Movie Review Sentiment Analysis API",
    description="API for analyzing sentiment in movie reviews",
    version="1.0.0",
)



# Initialize model service
model_service = ModelService()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Movie Review Sentiment Analysis API",
        "status": "active",
        "endpoints": {
            "/predict": "Predict sentiment for a single review",
            "/predict_batch": "Predict sentiment for multiple reviews",
            "/health": "Check API health status",
        },
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment for a single movie review
    """
    result = await model_service.predict_sentiment(request.text)
    return result


@app.post("/predict_batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(request: BatchReviewRequest):
    """
    Predict sentiment for multiple movie reviews
    """
    result = await model_service.predict_batch_sentiment(request.texts)
    return result


@app.get("/health")
async def health_check():
    """
    Check the health status of the API
    """
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "device": str(model_service.device),
    }


if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
