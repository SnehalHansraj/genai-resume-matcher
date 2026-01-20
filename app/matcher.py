import numpy as np
import faiss

class ResumeMatcher:
    def __init__(self, embeddings):
        self.embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def match(self, job_embedding):
        distances, indices = self.index.search(
            np.array([job_embedding]).astype("float32"), k=1
        )
        score = 1 / (1 + distances[0][0])
        return indices[0][0], score
