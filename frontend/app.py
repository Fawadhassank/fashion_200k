from flask import Flask, render_template, request
from pathlib import Path
import requests
import base64

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000/search-image"

# frontend/app.py se project root tak jana
FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent


def image_file_to_base64(image_path):
    """
    FastAPI returns image_path like:
    train/000000.jpg
    test/000002.jpg

    This function converts it to:
    D:/enjoy_till_it_lasts/train/000000.jpg
    """
    if not image_path:
        return None

    local_path = PROJECT_ROOT / image_path

    if not local_path.exists():
        print(f"[img lookup] image not found: {local_path}")
        return None

    try:
        with open(local_path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[img lookup] failed to read image {local_path}: {e}")
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query_image = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            results = [{"error": "No image file uploaded."}]
            return render_template("index.html", results=results, query_image=query_image)

        try:
            top_k = int(request.form.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5

        top_k = max(1, min(top_k, 5))

        # Show uploaded query image
        file_bytes = file.read()
        query_image = "data:image/jpeg;base64," + base64.b64encode(file_bytes).decode("utf-8")

        # Send same file bytes to FastAPI
        try:
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        file.filename,
                        file_bytes,
                        file.content_type or "image/jpeg"
                    )
                },
                data={"top_k": str(top_k)},
                timeout=120
            )
        except requests.exceptions.RequestException as e:
            print(f"[search request] failed to reach FastAPI backend: {e}")
            results = [{"error": f"Could not reach FastAPI backend: {e}"}]
            return render_template("index.html", results=results, query_image=query_image)

        if response.status_code != 200:
            print(f"[search request] FastAPI returned {response.status_code}: {response.text[:300]}")
            results = [{"error": f"FastAPI returned {response.status_code}"}]
            return render_template("index.html", results=results, query_image=query_image)

        data = response.json()

        if data.get("status") != "success":
            results = [{"error": data.get("message", "FastAPI returned an error.")}]
            return render_template("index.html", results=results, query_image=query_image)

        results = data.get("results", [])[:top_k]

        # Convert returned image_path to base64 for HTML display
        for r in results:
            r["img_data"] = image_file_to_base64(r.get("image_path"))

    return render_template("index.html", results=results, query_image=query_image)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)