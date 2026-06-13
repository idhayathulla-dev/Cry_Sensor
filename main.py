from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import HubertModel
import librosa
import json
import tempfile
import os
import urllib.request
from pydantic import BaseModel
from typing import List, Dict, Any

# -----------------------------
# GEMINI AI RECOMMENDATIONS & CHAT
# -----------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBfo5XqFONj1U-MM6ydOLts0zFLM0mG1dE")

def call_gemini(payload: dict) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return ""

def get_ai_suggestion(results: dict) -> str:
    top_cry = max(results.items(), key=lambda x: x[1])
    
    label_map = {
        "hungry": "Hungry",
        "tired": "Tired",
        "discomfort": "Discomfort",
        "burping": "Burping",
        "belly_pain": "Belly Pain"
    }
    
    top_label_display = label_map.get(top_cry[0], top_cry[0].replace('_', ' ').capitalize())
    
    other_cries = [
        f"{label_map.get(k, k.replace('_', ' ').capitalize())} ({v}%)"
        for k, v in results.items() if k != top_cry[0]
    ]
    
    prompt = (
        f"My baby is crying. An AI analysis suggests the most likely reason is {top_label_display} "
        f"with {top_cry[1]}% confidence. The other possibilities are {', '.join(other_cries)}. "
        f"Provide a short, actionable, and empathetic suggestion for the parent in 1-2 sentences. Do not use markdown."
    )
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    
    suggestion = call_gemini(payload)
    if not suggestion:
        suggestion = f"It looks like your baby might be {top_label_display.lower()}. Consider checking if they need to be fed, changed, or comforted."
    return suggestion


# -----------------------------
# MODEL
# -----------------------------

class HuBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )

        for param in self.hubert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 5)
        )

    def forward(self, input_values):
        outputs = self.hubert(input_values)

        pooled = outputs.last_hidden_state.mean(dim=1)

        logits = self.classifier(pooled)

        return logits


# -----------------------------
# APP
# -----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD LABELS
# -----------------------------

with open("label_encoder.json", "r") as f:
    labels = json.load(f)

classes = labels["classes"]

# -----------------------------
# LOAD MODEL
# -----------------------------

model = HuBERTClassifier()

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model Loaded Successfully")


# -----------------------------
# ROUTES
# -----------------------------

@app.get("/")
def home():
    return {"status": "CrySense API Running"}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await audio.read()
        tmp.write(contents)
        temp_path = tmp.name

    try:
        # Load audio
        waveform, sr = librosa.load(
            temp_path,
            sr=16000
        )

        # Convert to tensor
        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(waveform)
            probs = F.softmax(
                logits,
                dim=1
            )
            pred_idx = torch.argmax(
                probs,
                dim=1
            ).item()

        prediction = classes[pred_idx]
        scores = {}
        results_for_prompt = {}

        for i, label in enumerate(classes):
            val = float(probs[0][i])
            scores[label] = round(val, 4)
            results_for_prompt[label] = round(val * 100)

        suggestion = get_ai_suggestion(results_for_prompt)

        return {
            "prediction": prediction,
            "confidence": round(
                float(probs[0][pred_idx]),
                4
            ),
            "scores": scores,
            "suggestion": suggestion
        }
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

class ChatRequest(BaseModel):
    chat_history: List[Dict[str, Any]]

@app.post("/chat")
def chat(request: ChatRequest):
    system_instruction = (
        "You are CrySense Assistant, a specialized AI helper for the CrySense web app. "
        "Your primary goal is to provide accurate, empathetic, and safe information to parents and caregivers.\n"
        "**Persona:** Empathetic, Reassuring, Knowledgeable, Clear, Safe & Responsible.\n"
        "**Core Knowledge:** You are an expert on the CrySense app (AI cry analysis, data logging, pediatrician advice feature).\n"
        "**VERY IMPORTANT RULE:** NEVER provide medical advice. If asked for medical help, you MUST state your limitation and direct the user to a professional."
    )
    
    contents = [
        {"role": "model", "parts": [{"text": system_instruction}]}
    ]
    for msg in request.chat_history:
        contents.append({
            "role": msg.get("role"),
            "parts": [{"text": msg.get("parts", [{}])[0].get("text", "")}]
        })
        
    payload = {"contents": contents}
    response_text = call_gemini(payload)
    if not response_text:
        response_text = "Sorry, I'm having trouble connecting right now. Please try again later."
    return {"text": response_text}

@app.get("/health")
def health():
    return {
        "model_loaded": True,
        "classes": classes
    }