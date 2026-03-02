from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import numpy as np
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
from similarity import generate_embedding, cosine_sim, euclidean_sim, manhattan_sim

# ---------------- APP INIT ---------------- #

app = FastAPI(title="Cosine Similarity API")

# Enable CORS (Frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for graph image)
app.mount("/static", StaticFiles(directory="."), name="static")

# ---------------- HOME ROUTE ---------------- #

@app.get("/")
def home():
    return {"message": "Cosine Similarity API is running successfully 🚀"}

# ---------------- LOAD DATASET ---------------- #

with open("dataset.json", "r") as f:
    dataset = json.load(f)

dataset_embeddings = [generate_embedding(item["text"]) for item in dataset]

# ---------------- QUERY ROUTE ---------------- #

@app.post("/query")
async def query(
    input_type: str = Form(...),
    text: str = Form(None),
    file: UploadFile = File(None)
):

    # -------- HANDLE INPUT -------- #

    if input_type == "text":
        if not text:
            return {"error": "Text input required"}
        query_input = text

    elif input_type == "image":
        if not file:
            return {"error": "Image file required"}
        image = Image.open(io.BytesIO(await file.read()))
        query_input = "Image related to " + file.filename

    else:
        return {"error": "Invalid input_type. Use 'text' or 'image'"}

    # -------- GENERATE EMBEDDING -------- #

    query_embedding = generate_embedding(query_input)

    cosine_scores = []
    euclidean_scores = []
    manhattan_scores = []

    for emb in dataset_embeddings:
        cosine_scores.append(float(cosine_sim(query_embedding, emb)))
        euclidean_scores.append(float(euclidean_sim(query_embedding, emb)))
        manhattan_scores.append(float(manhattan_sim(query_embedding, emb)))

    avg_cosine = float(np.mean(cosine_scores))
    variance = float(np.var(cosine_scores))

    # -------- GENERATE GRAPH -------- #

    plt.figure()
    plt.bar(range(len(cosine_scores)), cosine_scores)
    plt.title("Cosine Similarity Scores")
    plt.xlabel("Dataset Index")
    plt.ylabel("Score")

    graph_path = "similarity_graph.png"
    plt.savefig(graph_path)
    plt.close()

    # -------- EXPLANATION -------- #

    explanation = (
        f"The average cosine similarity score is {round(avg_cosine,2)}. "
        f"The variance is {round(variance,4)}. "
        f"Cosine similarity measures angular distance between embeddings. "
        f"Higher values (closer to 1) indicate stronger semantic similarity."
    )

    # -------- STORE RESULT IN JSON -------- #

    result_data = {
        "query": query_input,
        "average_score": avg_cosine,
        "variance": variance,
        "cosine_scores": cosine_scores
    }

    with open("results.json", "a") as f:
        f.write(json.dumps(result_data) + "\n")

    # -------- RETURN RESPONSE -------- #

    return {
        "average_score": avg_cosine,
        "variance": variance,
        "cosine_scores": cosine_scores,
        "graph_url": f"http://localhost:8000/static/{graph_path}",
        "explanation": explanation
    }
