# fastapi_app.py

from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import faiss
import open_clip
import torch
from PIL import Image
import pickle

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Fashion Visual Search API")

# ----------------------------
# Load model and preprocessing
# IMPORTANT:
# Keep this model/pretrained same as the one used to create clip_embeddings.npy
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2B-s34B-b79K"
)

model.to(device)
model.eval()

# ----------------------------
# Load embeddings and metadata
# ----------------------------
embeddings = np.load("clip_embeddings.npy").astype("float32")

with open("clip_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Ensure embeddings and metadata length match
min_len = min(len(embeddings), len(metadata))
embeddings = embeddings[:min_len]
metadata = metadata[:min_len]

print("[startup] Device:", device)
print("[startup] Embeddings shape:", embeddings.shape)
print("[startup] Metadata records:", len(metadata))
print("[startup] Sample metadata row 0:", metadata[0])
print("[startup] Sample metadata row 1:", metadata[1])

# ----------------------------
# Build FAISS index
# Cosine similarity = normalized vectors + inner product
# ----------------------------
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("[startup] FAISS index total:", index.ntotal)

# ----------------------------
# Helper: Get CLIP embedding for uploaded image
# ----------------------------
def get_clip_embedding(img: Image.Image):
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


# ----------------------------
# Helper: Search similar images
# ----------------------------
def get_match_quality(score: float):
    if score >= 0.75:
        return "strong"
    elif score >= 0.65:
        return "medium"
    else:
        return "weak"


def build_explanation(row, score: float):
    category1 = row.get("category1", "")
    category2 = row.get("category2", "")
    category3 = row.get("category3", "")

    quality = get_match_quality(score)

    score_percent = round(score * 100)

    if quality == "strong":
        confidence_text = "a strong visual match"
    elif quality == "medium":
        confidence_text = "a moderate visual match"
    else:
        confidence_text = "the closest available match, but confidence is low"

    explanation = (
        f"This product was selected because its image embedding is close to the uploaded query image "
        f"in the CLIP visual search space. It appears to be {confidence_text} with a similarity score "
        f"of {score_percent}%. The retrieved catalogue item is categorized as '{category1} / {category2}', "
        f"and its product description is '{category3}', which indicates similar garment type, silhouette, "
        f"or visible style cues compared with the query image."
    )

    return explanation


def search_similar(query_emb, k=5):
    k = int(k)
    k = max(1, k)
    k = min(k, len(metadata))

    faiss.normalize_L2(query_emb)

    # Search slightly more so we can filter/return better results if needed
    search_k = min(max(k * 3, 15), len(metadata))

    scores, indices = index.search(query_emb, search_k)

    results = []

    for i in range(search_k):
        idx = int(indices[0][i])
        score = float(scores[0][i])

        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata[idx]
        quality = get_match_quality(score)

        result = {
            "rank": len(results) + 1,
            "category1": row.get("category1", ""),
            "category2": row.get("category2", ""),
            "category3": row.get("category3", ""),
            "item_id": row.get("item_id", ""),
            "image_path": row.get("image_path", ""),
            "score": score,
            "match_quality": quality,
            "explanation": build_explanation(row, score)
        }

        results.append(result)

        if len(results) == k:
            break

    return results


# ----------------------------
# API endpoint
# ----------------------------
@app.post("/search-image")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    try:
        image = Image.open(file.file).convert("RGB")

        query_emb = get_clip_embedding(image)

        results = search_similar(query_emb, k=top_k)

        return {
            "status": "success",
            "model": "ViT-B-32 / laion2B-s34B-b79K",
            "index_type": "FAISS IndexFlatIP with normalized CLIP embeddings",
            "index_size": len(metadata),
            "top_k": top_k,
            "results_returned": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    


# ----------------------------
# Health check endpoint
# ----------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Fashion Visual Search API is running"
    }