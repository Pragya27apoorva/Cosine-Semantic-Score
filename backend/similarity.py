import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean, cityblock

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text):
    return model.encode(text)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def euclidean_sim(a, b):
    return float(1 / (1 + euclidean(a, b)))

def manhattan_sim(a, b):
    return float(1 / (1 + cityblock(a, b)))
