import faiss
import numpy as np

class FaissService:
    def __init__(self, dim=512):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # L2 distance
        self.id_map = {}  # idx -> name
        self.counter = 0

    def add_embedding(self, embedding, name):
        embedding = np.array([embedding]).astype("float32")
        self.index.add(embedding)
        self.id_map[self.counter] = name
        self.counter += 1

    def search(self, embedding, threshold=0.55):
        embedding = np.array([embedding]).astype("float32")
        if self.index.ntotal == 0:
            return "Unknown", 1.0
        D, I = self.index.search(embedding, k=1)
        if D[0][0] < threshold:
            return self.id_map[I[0][0]], D[0][0]
        return "Unknown", D[0][0]
