import numpy as np
import os
from deepface import DeepFace

class RecognitionService:
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.face_memory = []

    def get_embedding(self, face):
        try:
            emb = DeepFace.represent(face, model_name="ArcFace", enforce_detection=False)
            return np.array(emb[0]["embedding"])
        except:
            return None

    def recognize(self, face):
        identity = "Unknown"
        embedding = self.get_embedding(face)

        if embedding is None:
            return identity

        best_dist, best_name = 999, None
        for mem in self.face_memory:
            dist = np.linalg.norm(embedding - mem["embedding"])
            if dist < best_dist:
                best_dist, best_name = dist, mem["name"]

        if best_dist < 0.55:
            return best_name

        try:
            result = DeepFace.find(
                face, self.faces_dir,
                model_name="ArcFace",
                enforce_detection=False
            )
            if len(result[0]) > 0:
                identity = os.path.basename(os.path.dirname(result[0].iloc[0]["identity"]))
                self.face_memory.append({"name": identity, "embedding": embedding})
        except:
            pass

        return identity
