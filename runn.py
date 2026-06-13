import torch

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=False
)

state_dict = checkpoint["model_state_dict"]

print(state_dict["classifier.0.weight"].shape)
print(state_dict["classifier.3.weight"].shape)
print(state_dict["classifier.6.weight"].shape)