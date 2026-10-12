import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_DATA_PATH, DENSE_EMBEDDINGS_PATH, DENSE_CHUNKS_PATH

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """Split text into overlapping word windows to avoid truncating long articles."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

class DenseRetriever:
    def __init__(self, model_name: str = 'all-mpnet-base-v2', chunk_size: int = 150, chunk_overlap: int = 30):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None
        self.df = None
        self.chunk_embeddings = None
        self.chunk_doc_ids = None
        
        # Load dataset
        if os.path.exists(PROCESSED_DATA_PATH):
            self.df = pd.read_csv(PROCESSED_DATA_PATH)
            self.df['text'] = self.df['text'].fillna("")
        else:
            raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run preprocess.py first.")
            
        self._initialize_model()
        
    def _initialize_model(self):
        """Load SentenceTransformer and compute/load chunk embeddings."""
        print(f"Loading SentenceTransformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        if os.path.exists(DENSE_EMBEDDINGS_PATH) and os.path.exists(DENSE_CHUNKS_PATH):
            print("Loading existing document chunk embeddings...")
            self.chunk_embeddings = np.load(DENSE_EMBEDDINGS_PATH)
            self.chunk_doc_ids = np.load(DENSE_CHUNKS_PATH)
        else:
            print("Chunking long articles and computing dense embeddings for all documents...")
            all_chunks = []
            chunk_doc_indices = []
            
            for doc_idx, text in enumerate(self.df['text'].tolist()):
                chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_doc_indices.append(doc_idx)
                    
            print(f"Total articles: {len(self.df)} | Total passages/chunks created: {len(all_chunks)}")
            self.chunk_doc_ids = np.array(chunk_doc_indices, dtype=np.int32)
            self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True, batch_size=64)
            
            # Save for future use
            np.save(DENSE_EMBEDDINGS_PATH, self.chunk_embeddings)
            np.save(DENSE_CHUNKS_PATH, self.chunk_doc_ids)
                
    def search(self, query: str, top_k: int = 10) -> list:
        """
        1. Encode incoming query
        2. Cosine similarity against all document chunks
        3. Max-pool chunk similarities back to document level
        4. Return top K documents
        """
        query_emb = self.model.encode([query])
        
        # Calculate cosine similarity with all chunks
        chunk_sims = cosine_similarity(query_emb, self.chunk_embeddings).flatten()
        
        # Aggregate chunk-level similarities to document-level scores via Max Pooling
        doc_scores = np.full(len(self.df), -1.0, dtype=float)
        np.maximum.at(doc_scores, self.chunk_doc_ids, chunk_sims)
        
        # Get top K indices
        top_indices = doc_scores.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(doc_scores[idx])
            
            doc = self.df.iloc[idx]
            results.append({
                "document_id": doc["document_id"],
                "score": score,
                "text": doc["text"],
                "tags": doc["tags"],
                "rank": rank
            })
            
        return results

