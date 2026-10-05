from src.retrievers.tfidf_retriever import TFIDFRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever

class RRFRetriever:
    def __init__(self, k: int = 60):
        self.k = k
        self.tfidf = TFIDFRetriever()
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        
    def search(self, query: str, top_k: int = 10) -> list:
        """
        1. Retrieve top K from each model (we retrieve more, say 50, to get good overlap)
        2. Assign rank positions
        3. Calculate RRF score: 1 / (k + rank)
        4. Aggregate scores
        5. Return final top K
        """
        # Retrieve more candidates internally to allow for meaningful rank fusion
        candidate_pool_size = max(50, top_k * 2)
        
        tfidf_res = self.tfidf.search(query, top_k=candidate_pool_size)
        bm25_res = self.bm25.search(query, top_k=candidate_pool_size)
        dense_res = self.dense.search(query, top_k=candidate_pool_size)
        
        # Combine documents from all rankers
        doc_scores = {} # dict mapping doc_id -> (rrf_score, text, tags)
        
        def add_to_rrf(results_list):
            for res in results_list:
                doc_id = res["document_id"]
                rank = res["rank"]
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "score": 0.0,
                        "text": res["text"],
                        "tags": res["tags"],
                        "document_id": doc_id
                    }
                
                doc_scores[doc_id]["score"] += 1.0 / (self.k + rank)
                
        add_to_rrf(tfidf_res)
        add_to_rrf(bm25_res)
        add_to_rrf(dense_res)
        
        # Sort descending by RRF score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Format top K
        final_results = []
        for rank, doc in enumerate(sorted_docs[:top_k], start=1):
            final_results.append({
                "document_id": doc["document_id"],
                "score": doc["score"],
                "text": doc["text"],
                "tags": doc["tags"],
                "rank": rank
            })
            
        return final_results
