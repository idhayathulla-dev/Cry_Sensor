from model import HuBERTClassifier
import torch

model = HuBERTClassifier()

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("SUCCESS")