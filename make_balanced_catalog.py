# make_balanced_catalog.py

import pandas as pd

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Combine train + test because test_data has many tops
df = pd.concat([train_df, test_df], ignore_index=True)

# Remove duplicate item/image rows
df = df.drop_duplicates(subset=["image_path", "item_ID"])

print("Before balancing:")
print(df["category1"].value_counts())

balanced_parts = []

SAMPLE_PER_CATEGORY = 5000

for category in ["dresses", "skirts", "pants", "jackets", "tops"]:
    part = df[df["category1"] == category]

    if len(part) > SAMPLE_PER_CATEGORY:
        part = part.sample(SAMPLE_PER_CATEGORY, random_state=42)

    balanced_parts.append(part)

balanced_df = pd.concat(balanced_parts, ignore_index=True)

balanced_df.to_csv("balanced_catalog.csv", index=False)

print("\nBalanced catalog created: balanced_catalog.csv")
print("Rows:", len(balanced_df))
print(balanced_df["category1"].value_counts())
print("\nCategory2 counts:")
print(balanced_df["category2"].value_counts().head(30))