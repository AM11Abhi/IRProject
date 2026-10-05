import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_DATA_PATH, DENSE_EMBEDDINGS_PATH

class DenseRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.df = None
        self.embeddings = None
        
        # Load dataset
        if os.path.exists(PROCESSED_DATA_PATH):
            self.df = pd.read_csv(PROCESSED_DATA_PATH)
            self.df['text'] = self.df['text'].fillna("")
        else:
            raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run preprocess.py first.")
            
        self._initialize_model()
        
    def _initialize_model(self):
        """Load SentenceTransformer and compute/load embeddings."""
        print(f"Loading SentenceTransformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        if os.path.exists(DENSE_EMBEDDINGS_PATH):
            print("Loading existing document embeddings...")
            self.embeddings = np.load(DENSE_EMBEDDINGS_PATH)
        else:
            print("Computing dense embeddings for all documents. This might take a few minutes...")
            # We encode the raw 'text' rather than 'clean_text' because sentence transformers 
            # generally perform better on natural, un-preprocessed sentences with full context.
            texts_to_encode = self.df['text'].tolist()
            self.embeddings = self.model.encode(texts_to_encode, show_progress_bar=True)
            
            # Save for future use
            np.save(DENSE_EMBEDDINGS_PATH, self.embeddings)
                
    def search(self, query: str, top_k: int = 10) -> list:
        """
        1. Encode incoming query
        2. Cosine similarity
        3. Return top K
        """
        # Encode natural query (no stopword removal/lowercasing needed for dense model)
        query_emb = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_emb, self.embeddings).flatten()
        
        # Get top K indices
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            
            doc = self.df.iloc[idx]
            results.append({
                "document_id": doc["document_id"],
                "score": score,
                "text": doc["text"],
                "tags": doc["tags"],
                "rank": rank
            })
            
        return results
