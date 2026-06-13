import torch
import torch.nn as nn
from transformers import HubertModel


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