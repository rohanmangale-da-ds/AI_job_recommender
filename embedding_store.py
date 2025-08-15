import faiss
import numpy as np
# from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings


class JobVectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Local embedding model — no API key required
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.index = None
        self.job_data = None

    def create_store(self, job_df: pd.DataFrame):
        """Create FAISS vector store from job descriptions."""
        texts = job_df["description"].tolist()
        vectors = self.embeddings.embed_documents(texts)
        vectors_np = np.array(vectors).astype("float32")

        self.index = faiss.IndexFlatL2(vectors_np.shape[1])
        self.index.add(vectors_np)
        self.job_data = job_df

    def search(self, query: str, k: int = 5):
        """Search similar jobs for the query."""
        query_vector = np.array([self.embeddings.embed_query(query)]).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return self.job_data.iloc[indices[0]].to_dict(orient="records")

