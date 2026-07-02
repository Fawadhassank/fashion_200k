# rebuild_clip_metadata.py

import pandas as pd
import pickle
import shutil
from pathlib import Path
import numpy as np

EMBEDDINGS_PATH = Path("clip_embeddings.npy")
METADATA_PATH = Path("clip_metadata.pkl")

csv_candidates = [
    Path("train_data.csv"),
    Path("test_data.csv"),
    Path("train_data.xlsx"),
    Path("test_data.xlsx"),
]

embeddings = np.load(EMBEDDINGS_PATH)
embedding_rows = embeddings.shape[0]

print("Embeddings rows:", embedding_rows)

selected_file = None
selected_df = None

for file_path in csv_candidates:
    if not file_path.exists():
        continue

    print("\nChecking:", file_path)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if len(df) == embedding_rows:
        selected_file = file_path
        selected_df = df
        break

if selected_df is None:
    raise ValueError(
        "No CSV/Excel file rows matched embeddings rows. "
        "This means embeddings may have been generated from another file "
        "or embeddings need to be rebuilt."
    )

print("\nSelected metadata source:", selected_file)

required_columns = ["image_path", "category1", "category2", "category3", "item_ID"]

missing = [col for col in required_columns if col not in selected_df.columns]
if missing:
    raise ValueError(f"Missing columns in selected file: {missing}")

if METADATA_PATH.exists():
    backup_path = Path("clip_metadata_old_wrong.pkl")
    shutil.copy(METADATA_PATH, backup_path)
    print("Old metadata backup created:", backup_path)

metadata = []

for _, row in selected_df.iterrows():
    metadata.append({
        "image_path": str(row["image_path"]),
        "category1": str(row["category1"]),
        "category2": str(row["category2"]),
        "category3": str(row["category3"]),
        "item_id": str(row["item_ID"])
    })

with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("\nNew clip_metadata.pkl created successfully.")
print("Metadata records:", len(metadata))
print("\nFirst 5 records:")
for item in metadata[:5]:
    print(item)