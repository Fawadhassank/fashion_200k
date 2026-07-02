import pickle
import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

CSV_PATH = Path("balanced_catalog.csv")

EMBEDDINGS_PATH = Path("clip_embeddings.npy")
METADATA_PATH = Path("clip_metadata.pkl")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2B-s34B-b79K"
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

df = pd.read_csv(CSV_PATH)

required_columns = ["image_path", "category1", "category2", "category3", "item_ID"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

print("CSV file:", CSV_PATH)
print("CSV rows:", len(df))
print("Category1 counts:")
print(df["category1"].value_counts())

# Backup old files
for old_file in [EMBEDDINGS_PATH, METADATA_PATH]:
    if old_file.exists():
        backup_file = old_file.with_suffix(old_file.suffix + ".backup")
        shutil.copy(old_file, backup_file)
        print("Backup created:", backup_file)

# Build local image lookup by filename
print("\nScanning local image files...")
image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

file_map = {}
for ext in image_extensions:
    for path in Path(".").rglob(f"*{ext}"):
        file_map[path.name] = path

print("Local images found:", len(file_map))

def resolve_image_path(csv_image_path):
    p = Path(str(csv_image_path))

    # If direct path exists
    if p.exists():
        return p

    # If old path from colab, use filename
    filename = p.name

    if filename in file_map:
        return file_map[filename]

    # fallback by train/test folder guess
    train_guess = Path("train") / filename
    test_guess = Path("test") / filename

    if train_guess.exists():
        return train_guess
    if test_guess.exists():
        return test_guess

    return None

print("\nLoading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)
model = model.to(device)
model.eval()

all_embeddings = []
metadata = []

batch_images = []
batch_metadata = []

missing_images = 0
bad_images = 0

print("\nGenerating embeddings...")

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = resolve_image_path(row["image_path"])

        if image_path is None:
            missing_images += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image)
        except Exception:
            bad_images += 1
            continue

        batch_images.append(image_tensor)

        batch_metadata.append({
            "image_path": str(image_path).replace("\\", "/"),
            "category1": str(row["category1"]),
            "category2": str(row["category2"]),
            "category3": str(row["category3"]),
            "item_id": str(row["item_ID"])
        })

        if len(batch_images) == BATCH_SIZE:
            images_tensor = torch.stack(batch_images).to(device)

            features = model.encode_image(images_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy().astype("float32"))
            metadata.extend(batch_metadata)

            batch_images = []
            batch_metadata = []

    if batch_images:
        images_tensor = torch.stack(batch_images).to(device)

        features = model.encode_image(images_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy().astype("float32"))
        metadata.extend(batch_metadata)

embeddings = np.vstack(all_embeddings).astype("float32")

print("\nMissing images:", missing_images)
print("Bad images:", bad_images)
print("Final embeddings shape:", embeddings.shape)
print("Final metadata records:", len(metadata))

if embeddings.shape[0] != len(metadata):
    raise ValueError("Embeddings and metadata count mismatch!")

np.save(EMBEDDINGS_PATH, embeddings)
print("Saved:", EMBEDDINGS_PATH)

with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("Saved:", METADATA_PATH)

print("\nFirst 5 metadata records:")
for item in metadata[:5]:
    print(item)

print("\nDone.")