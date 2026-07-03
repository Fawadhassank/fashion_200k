# Architecture Document — Fashion Visual Search

## 1. Overview

This project implements a fashion visual search prototype using the Fashion-200K dataset as a proxy for an e-commerce product catalogue. The system allows a user to upload an image of a garment and returns a ranked list of visually similar catalogue products.

The primary goal is to demonstrate an end-to-end retrieval system rather than train a new deep learning model from scratch. The final system includes image embedding generation, metadata preparation, FAISS indexing, an HTTP search endpoint, a simple frontend, and explainable ranked results.

---

## 2. System flow

The runtime flow is:

```text
User uploads image
    ↓
FastAPI receives image
    ↓
OpenCLIP converts image to vector embedding
    ↓
Embedding is normalized
    ↓
FAISS searches nearest catalogue embeddings
    ↓
Metadata is joined with FAISS results
    ↓
API returns ranked results with explanation
```

The Flask frontend is a demonstration layer on top of the FastAPI backend. The main system interface is the HTTP API.

---

## 3. Model choice

The embedding model used is:

```text
OpenCLIP ViT-B-32 / laion2B-s34B-b79K
```

CLIP-style models are suitable for this task because they generate general-purpose visual embeddings that capture semantic and visual similarity. This allows the system to retrieve visually similar products without supervised training on every possible fashion category.

The same model must be used for both offline catalogue embedding generation and online query embedding. During development, model mismatch was tested and corrected. The saved embeddings matched the `laion2B-s34B-b79K` checkpoint, so the FastAPI service uses the same checkpoint.

---

## 4. Catalogue and metadata preparation

The original dataset was split across `train_data.csv` and `test_data.csv`. During testing, the initial FAISS index contained only dresses and jackets. This caused shirt queries to return jackets, because shirts were not present in the searchable index.

To fix this, a balanced catalogue was created using:

```text
balanced_catalog.csv
```

It contains 25,000 indexed images:

```text
dresses = 5000
skirts  = 5000
pants   = 5000
jackets = 5000
tops    = 5000
```

This improved retrieval quality for shirts and tops because the searchable catalogue now includes the relevant categories.

Each metadata record contains:

```text
image_path
category1
category2
category3
item_id
```

The `item_id` bug was fixed by converting the metadata from a batched tuple format into a one-record-per-image dictionary format. This ensures each FAISS result index maps correctly to its product metadata.

---

## 5. Embedding generation

Catalogue embeddings are generated offline using OpenCLIP.

For each catalogue image:

1. Load image using Pillow.
2. Convert image to RGB.
3. Apply OpenCLIP preprocessing.
4. Generate image embedding.
5. Normalize the embedding.
6. Save all embeddings to `clip_embeddings.npy`.
7. Save aligned metadata to `clip_metadata.pkl`.

The one-to-one alignment between embeddings and metadata is critical. The row index of an embedding must match the row index of the metadata item.

---

## 6. Indexing strategy

The prototype uses:

```text
FAISS IndexFlatIP
```

All embeddings are L2-normalized, so inner product search behaves like cosine similarity.

This index is accurate and simple for a 25,000-image prototype. It is not the best production choice for millions of images because flat search compares the query against every vector.

For a production system with around 2 million images, a more scalable index should be used:

```text
FAISS IVF
FAISS HNSW
FAISS IVFPQ
Vector database with ANN search
```

These approximate nearest-neighbor methods reduce latency and memory cost while maintaining acceptable recall.

---

## 7. Serving architecture

The FastAPI service loads the following once at startup:

```text
OpenCLIP model
CLIP preprocessing pipeline
clip_embeddings.npy
clip_metadata.pkl
FAISS index
```

At request time, the service only needs to:

1. Read uploaded image.
2. Generate query embedding.
3. Normalize query embedding.
4. Search FAISS.
5. Build JSON response.

The endpoint is:

```text
POST /search-image
```

The response includes:

```text
rank
category1
category2
category3
item_id
image_path
score
match_quality
explanation
```

---

## 8. Explanation strategy

Each returned product includes a plain-text explanation.

The explanation is generated locally and does not rely on a third-party inference API. It is based on:

- CLIP similarity score
- Match quality label
- Retrieved category metadata
- Product description field

Example explanation:

```text
This product was selected because its image embedding is close to the uploaded query image in the CLIP visual search space. It appears to be a strong visual match with a similarity score of 83%. The retrieved catalogue item is categorized as 'tops / long sleeved tops', and its product description is 'red checked long sleeve top', which indicates similar garment type, silhouette, or visible style cues compared with the query image.
```

This explanation is designed for non-technical stakeholders. It explains the retrieval decision without exposing implementation details that would be hard for users to understand.

A stronger production version would add local visual attribute extraction for color, pattern, sleeve length, garment type, and texture. Those detected attributes could then make explanations more specific.

---

## 9. Match quality

Similarity scores are mapped into simple quality labels:

```text
strong
medium
weak
```

This helps avoid presenting low-confidence results as equally reliable. For example, if the nearest available result is below the preferred threshold, the system can still return it but mark it as weak.

This is important because visual search systems may always return nearest neighbors even when the catalogue does not contain a truly close match.

---

## 10. Frontend

The Flask frontend is included for demonstration and manual testing.

It allows users to:

- Upload query image.
- Choose number of results.
- View returned product images.
- See item ID and categories.
- View similarity score and match quality.
- Read the explanation for each result.

The frontend calls the FastAPI backend and renders returned image paths as local base64 images.

---

## 11. Current limitations

The current system is a working prototype with the following limitations:

1. It uses a balanced 25,000-image subset rather than the full dataset.
2. Inference currently runs on CPU unless CUDA is available.
3. The FAISS index is rebuilt in memory at service startup.
4. The explanation is deterministic and metadata-based.
5. The system does not yet perform dedicated color, pattern, sleeve, or texture detection.
6. Search quality depends on the coverage of categories in the indexed catalogue.
7. The current prototype uses exact FAISS search, which is not ideal for millions of images.

---

## 12. Production scaling plan

For a production catalogue of 2 million product images across hundreds of categories, I would make the following changes.

### Offline embedding pipeline

Embeddings should be generated in batch jobs using GPU workers. The pipeline should process new and updated catalogue images incrementally.

### Versioned index builds

Each index build should be versioned. The system should be able to roll back to a previous index if a new index has quality or metadata issues.

### Approximate nearest-neighbor indexing

Use FAISS IVF, HNSW, or IVFPQ to reduce memory usage and improve latency.

### Metadata store

Store product metadata in a database or key-value store. The vector index should return IDs, and the service should fetch metadata by ID.

### GPU query inference

Use GPU inference for low-latency query embedding generation.

### Category-aware reranking

After initial FAISS retrieval, apply reranking using category, color, pattern, or product-type signals. This reduces cases where visually similar but semantically wrong items appear.

### Monitoring

Track:

```text
p50 latency
p95 latency
error rate
index size
embedding generation failures
empty/weak match rate
top result score distribution
```

### Load testing

Test concurrent uploads and measure latency under realistic request load.

---

## 13. What I would improve next

Given more time, I would:

1. Build the full Fashion-200K index instead of a balanced subset.
2. Add local visual attribute extraction for better explanations.
3. Add category-aware reranking.
4. Add ANN indexing for larger-scale retrieval.
5. Add automated tests for metadata alignment.
6. Add a proper Dockerfile and deployment instructions.
7. Add load testing results using Locust or k6.

---

## 14. Conclusion

The project demonstrates a complete visual search pipeline: data preparation, embedding generation, FAISS indexing, API serving, frontend display, and explainable ranked results. The main engineering lesson was that retrieval quality depends not only on the model but also on correct metadata alignment and representative index coverage. Once the index was rebuilt with balanced categories, the system began returning relevant shirt/top results instead of unrelated jacket results.
