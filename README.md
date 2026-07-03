# Fashion Visual Search

A visual search prototype for fashion catalogue images using **OpenCLIP + FAISS + FastAPI**.

The system accepts an uploaded garment image and returns a ranked list of visually similar catalogue products. Each result includes product metadata, similarity score, match quality, local image path, and a plain-text explanation of why the product was selected.

---

## 1. What this project does

This project uses the Fashion-200K dataset as a proxy catalogue for an e-commerce visual search system.

Flow:

```text
Uploaded image
    ↓
OpenCLIP image embedding
    ↓
FAISS nearest-neighbor search
    ↓
Ranked product results
    ↓
Plain-text explanation per result
```

The backend exposes an HTTP endpoint:

```text
POST /search-image
```

The request accepts an image file and returns visually similar products.

---

## 2. Tech stack

- Python
- FastAPI
- Flask
- OpenCLIP
- FAISS
- PyTorch
- Pandas
- NumPy
- Pillow

Model used:

```text
OpenCLIP ViT-B-32 / laion2B-s34B-b79K
```

Vector index:

```text
FAISS IndexFlatIP with normalized CLIP embeddings
```

This gives cosine-similarity-style search.

---

## 3. Project structure

Recommended structure:

```text
.
├── fastapi_app.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── FashionSearch.postman_collection.json
├── make_balanced_catalog.py
├── rebuild_clip_embeddings_metadata_faiss.py
├── fix_metadata_image_paths.py
├── check_index_categories.py
├── train_data.csv
├── test_data.csv
├── balanced_catalog.csv
├── clip_embeddings.npy
├── clip_metadata.pkl
├── train/
│   └── image files
├── test/
│   └── image files
└── frontend/
    ├── app.py
    └── templates/
        └── index.html
```

---

## 4. Environment setup

### Windows PowerShell

Create virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install manually:

```powershell
pip install fastapi uvicorn python-multipart flask requests pandas numpy pillow torch torchvision faiss-cpu open-clip-torch tqdm
```

---

## 5. Data preparation

The project expects Fashion-200K image folders and CSV files:

```text
train/
test/
train_data.csv
test_data.csv
```

The CSV files should contain these columns:

```text
image_path
category1
category2
category3
item_ID
```

A balanced catalogue is created to avoid category imbalance. This is important because an earlier index containing only dresses and jackets caused shirt queries to return jacket results.

Create balanced catalogue:

```powershell
python make_balanced_catalog.py
```

This creates:

```text
balanced_catalog.csv
```

The balanced catalogue contains 25,000 images:

```text
dresses = 5000
skirts  = 5000
pants   = 5000
jackets = 5000
tops    = 5000
```

---

## 6. Build embeddings and metadata

Run:

```powershell
python rebuild_clip_embeddings_metadata_faiss.py
```

This creates:

```text
clip_embeddings.npy
clip_metadata.pkl
```

The metadata file stores one dictionary per indexed image:

```python
{
    "image_path": "train/000000.jpg",
    "category1": "tops",
    "category2": "long sleeved tops",
    "category3": "red checked long sleeve top",
    "item_id": "87459924_1"
}
```

If image paths are still old Colab-style paths, run:

```powershell
python fix_metadata_image_paths.py
```

Expected path format:

```text
train/000000.jpg
test/000001.jpg
```

---

## 7. Verify indexed categories

Run:

```powershell
python check_index_categories.py
```

Expected output should include all major categories:

```text
dresses
skirts
pants
jackets
tops
```

This confirms that the FAISS search catalogue is not limited to only one or two categories.

---

## 8. Run FastAPI backend

Start the backend:

```powershell
python -m uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --log-level debug
```

Health check:

```text
http://127.0.0.1:8000/
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Expected health response:

```json
{
  "status": "running",
  "message": "Fashion Visual Search API is running"
}
```

---

## 9. API endpoint

### POST `/search-image`

URL:

```text
http://127.0.0.1:8000/search-image
```

Request type:

```text
form-data
```

Fields:

| Field | Type | Description |
|---|---|---|
| file | File | Query image |
| top_k | Text/Number | Number of results to return |

Example request in Postman:

```text
file  = train/000000.jpg
top_k = 5
```

Example response:

```json
{
  "status": "success",
  "model": "ViT-B-32 / laion2B-s34B-b79K",
  "index_type": "FAISS IndexFlatIP with normalized CLIP embeddings",
  "index_size": 25000,
  "top_k": 1,
  "results_returned": 1,
  "results": [
    {
      "rank": 1,
      "category1": "tops",
      "category2": "long sleeved tops",
      "category3": "red maison kitsune women's check voile long sleeve top",
      "item_id": "87459924_1",
      "image_path": "test/000001.jpg",
      "score": 0.8257438540458679,
      "match_quality": "strong",
      "explanation": "This product was selected because its image embedding is close to the uploaded query image in the CLIP visual search space..."
    }
  ]
}
```

---

## 10. Run Flask frontend

Open a second terminal and activate the same virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Run frontend:

```powershell
python frontend/app.py
```

Open:

```text
http://127.0.0.1:5000/
```

Upload a garment image and the frontend will display:

- Query image
- Similar product cards
- Product image
- Category information
- Item ID
- Similarity score
- Match quality
- Explanation

---

## 11. Postman collection

Import this file into Postman:

```text
FashionSearch.postman_collection.json
```

It contains:

```text
GET  /
POST /search-image
```

Before running the request, ensure FastAPI is running on:

```text
http://127.0.0.1:8000
```

For the file field, manually select a local image if Postman does not preserve the file path after import.

---

## 12. Match quality logic

Current match quality labels:

```text
strong = high similarity
medium = moderate similarity
weak   = low similarity
```

The labels are calculated from the similarity score and are intended to help non-technical users understand result confidence.

---

## 13. Known limitations

- This prototype uses a balanced 25,000-image subset instead of the full Fashion-200K dataset to keep CPU-based development practical.
- Search quality depends on the categories present in the FAISS index.
- The explanation is deterministic and based on similarity score plus returned catalogue metadata.
- The current explanation does not perform separate color/pattern/sleeve detection.
- For production, the index should be built offline on GPU and served with approximate FAISS indexing.

---

## 14. Production considerations

For a production catalogue of around 2 million images:

- Generate embeddings offline using GPU batch jobs.
- Use a versioned embedding pipeline.
- Use FAISS IVF, HNSW, or IVFPQ instead of flat search.
- Store embeddings and metadata in object storage or a vector database.
- Use GPU inference for query image embedding.
- Add category-aware reranking.
- Add monitoring for latency, throughput, errors, and index freshness.
- Add load testing for concurrent image uploads.
- Add scheduled index refresh for new products.

---

## 15. Final run checklist

```text
[ ] Activate virtual environment
[ ] Install requirements
[ ] Build balanced_catalog.csv
[ ] Build clip_embeddings.npy and clip_metadata.pkl
[ ] Verify index categories
[ ] Start FastAPI backend
[ ] Test Postman /search-image
[ ] Start Flask frontend
[ ] Upload image and verify results
```
