---
tags:
- audio-classification
- baby-cry-detection
- pytorch
- hubert
language:
- en
library_name: transformers
pipeline_tag: audio-classification
---

# Baby Cry Classification Model

Classifies baby cries into 5 categories to help parents understand their baby's needs.

## Categories
- 🍼 **Hungry**: Baby needs feeding
- 😴 **Tired**: Baby needs sleep
- 😣 **Belly Pain**: Digestive discomfort
- 💨 **Burping**: Needs burping
- 😢 **Discomfort**: General discomfort (diaper, temperature, etc.)

## Model Details
- **Base Model**: HuBERT (facebook/hubert-base-ls960)
- **Architecture**: HuBERT + Custom Classification Head
- **Sample Rate**: 16kHz
- **Input Duration**: 5 seconds
- **Framework**: PyTorch + Transformers

## Usage
```python
import requests
import numpy as np

# Load audio file
audio_data = open("baby_cry.wav", "rb").read()

# Call API
API_URL = "https://api-inference.huggingface.co/models/dontcryai/baby-cry-classifier"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

response = requests.post(API_URL, headers=headers, data=audio_data)
result = response.json()

print(result)
# [{'label': 'hungry', 'score': 0.85}, ...]
```

## Training
Trained on custom baby cry dataset with data augmentation.
