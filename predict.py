import json
import torch
import librosa
import torch.nn.functional as F

from model import HuBERTClassifier

# load labels
with open("label_encoder.json") as f:
    labels = json.load(f)

# load model
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

# load audio
audio, sr = librosa.load(
    "hungryy.wav",
    sr=16000
)

audio = torch.tensor(audio).unsqueeze(0)

with torch.no_grad():
    logits = model(audio)

probs = F.softmax(logits, dim=1)

pred = torch.argmax(probs, dim=1).item()

print("Prediction:", labels["classes"][pred])
print("Confidence:", probs[0][pred].item())