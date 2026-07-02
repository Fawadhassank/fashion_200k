# check_metadata.py

import pickle
from collections import Counter
from pathlib import Path

METADATA_PATH = Path("clip_metadata.pkl")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Total records:", len(metadata))
print("Type:", type(metadata))

print("\nFirst 10 metadata records:")
for i, item in enumerate(metadata[:10]):
    print(f"\n--- Record {i} ---")
    print("Type:", type(item))
    print(item)

# Check if metadata is fixed dictionary format
if len(metadata) > 0 and isinstance(metadata[0], dict):
    print("\nMetadata format: DICTIONARY FORMAT")

    item_ids = [
        str(x.get("item_id", ""))
        for x in metadata
        if isinstance(x, dict)
    ]

    print("\nTotal item_id values:", len(item_ids))
    print("Unique item_id values:", len(set(item_ids)))

    print("\nTop repeated item_id values:")
    for value, count in Counter(item_ids).most_common(20):
        print(value, "=>", count)

    same_as_category = 0

    for x in metadata:
        if not isinstance(x, dict):
            continue

        item_id = str(x.get("item_id", ""))

        categories = [
            str(x.get("category1", "")),
            str(x.get("category2", "")),
            str(x.get("category3", "")),
            str(x.get("category", "")),
        ]

        if item_id in categories:
            same_as_category += 1

    print("\nRecords where item_id is same as category:", same_as_category)

    print("\nFirst 10 item_id values:")
    for x in metadata[:10]:
        print(x.get("item_id"))

else:
    print("\nMetadata format: NOT DICTIONARY FORMAT")
    print("This means clip_metadata.pkl is still wrong or batched.")

    print("\nTuple/list pattern check:")
    for i, item in enumerate(metadata[:12]):
        print(f"Record {i}: type={type(item)}, length={len(item) if hasattr(item, '__len__') else 'N/A'}")