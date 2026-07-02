# detect_clip_model_match.py

import pickle
import numpy as np
import torch
import open_clip
from PIL import Image
from pathlib import Path

EMBEDDINGS_PATH = "clip_embeddings.npy"
METADATA_PATH = "clip_metadata.pkl"

candidates = [
    ("ViT-B-32", "openai"),
    ("ViT-B-32", "laion2B-s34B-b79K"),
]

embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# Test first indexed image
idx = 0
stored_emb = embeddings[idx]
stored_emb = stored_emb / np.linalg.norm(stored_emb)

image_path = Path(metadata[idx]["image_path"])
print("Testing image:", image_path)
print("Expected item_id:", metadata[idx]["item_id"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

for model_name, pretrained in candidates:
    print("\nTesting:", model_name, pretrained)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )

    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model.encode_image(image_tensor)

    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy()[0].astype("float32")

    similarity = float(np.dot(query_emb, stored_emb))

    print("Similarity with stored embedding:", similarity)