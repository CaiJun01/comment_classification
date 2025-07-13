from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import json
from transformers import pipeline
import re
from contextlib import asynccontextmanager
from fastapi import Request

# Define input data schema
class Comment(BaseModel):
    text: str

# Function to perform NLI
def get_nli_label(premise, hypothesis, deberta_tokenizer, deberta_model):
    inputs = deberta_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = deberta_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)

    print(probs)

    # Instead of using argmax:
    if probs[0][0] > 0.4:  # contradiction
        label = "contradiction"
        confidence = probs[0][0]
    elif probs[0][2] > 0.005:  # entailment score
        label = "entailment"
        confidence = probs[0][2]
    else:
        label = "neutral"
        confidence = probs[0][1]

    # label_map = ['contradiction', 'neutral', 'entailment']
    # label = label_map[torch.argmax(probs).item()]
    # confidence = probs.max().item()

    return label, confidence

model = None
tokenizer = None
label_map = None
book_chunks = None
book_embeddings = None
deberta_tokenizer = None
deberta_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer
    app.state.model_path = "./debertav3base"   # update to your checkpoint path

    app.state.tokenizer = AutoTokenizer.from_pretrained("./debertav3base")
    app.state.model = AutoModelForSequenceClassification.from_pretrained("./debertav3base")
    app.state.model.eval()

    app.state.label_map = {0: "opinion", 1: "claim"}

    #load the book_embeddings
    loaded_embeddings = np.load("./book_embeddings.npy")
    app.state.book_embeddings = torch.tensor(loaded_embeddings)
    # app.state.book_embeddings = book_embeddings.cpu()

    #load the book_chunks
    with open("book_chunks.json", "r") as f:
        app.state.book_chunks = json.load(f)

    #load tokenizer to tell true or false
    app.state.deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
    app.state.deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-xlarge-mnli")

    yield

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Allow all origins (for development only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(comment: Comment, request: Request):
    # Split by punctuation and conjunctions
    segments = re.split(r'[.!?]|\bor\b', comment.text)
    segments = [s.strip() for s in segments if s.strip()]

    res = {}

    for com in segments:
        # Tokenize input
        inputs = request.app.state.tokenizer(
            com,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        # Run model
        with torch.no_grad():
            outputs = request.app.state.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        # Map label   
        label = request.app.state.label_map[prediction]

        if label == 'claim':
            #encode claim with embedder
            embedder = SentenceTransformer("BAAI/bge-base-en")
            claim_embedding = embedder.encode(com, convert_to_tensor=True)

            claim_embedding = claim_embedding.cpu()
            
            # Compute similarity with all chunks
            cosine_scores = util.cos_sim(claim_embedding, request.app.state.book_embeddings)

            # Find the best matching chunk
            top_index = cosine_scores.argmax().item()

            best_chunk = request.app.state.book_chunks[top_index]

            premise = best_chunk
            hypothesis = com

            result, confidence = get_nli_label(premise, hypothesis, request.app.state.deberta_tokenizer, request.app.state.deberta_model)
            print(f"Label: {label}, Confidence: {confidence}")

            # nli_pipeline = pipeline("text-classification", model="roberta-large-mnli")

            # result = nli_pipeline(f"{premise} </s> {hypothesis}")
            
            if result == "entailment":
                v = 'True'
            
            elif result == "contradiction":
                v = 'False'

            elif result == "neutral":
                v = 'Unknown'

            res[com] =  {
                "Claim or Opinion": label,
                "True or False": v,
                "Score": confidence,
                "Evidence": best_chunk,
                "Similarity Score": cosine_scores[0, top_index].item()
                }

        else:
            res[com] = {
                "Claim or Opinion": label,
                "confidence": round(confidence, 4)
            }

    return res