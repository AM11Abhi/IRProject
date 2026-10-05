from src.retrievers.tfidf_retriever import TFIDFRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.rrf_retriever import RRFRetriever

# Lazy loading of retrievers to save memory/time if not all are needed
_RETRIEVERS = {}

def get_retriever(model_name: str):
    if model_name not in _RETRIEVERS:
        if model_name == "tfidf":
            _RETRIEVERS[model_name] = TFIDFRetriever()
        elif model_name == "bm25":
            _RETRIEVERS[model_name] = BM25Retriever()
        elif model_name == "dense":
            _RETRIEVERS[model_name] = DenseRetriever()
        elif model_name == "rrf":
            _RETRIEVERS[model_name] = RRFRetriever()
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    return _RETRIEVERS[model_name]

def search(query: str, model: str = "tfidf", top_k: int = 10) -> list:
    """
    Common search interface.
    Supported models: 'tfidf', 'bm25', 'dense', 'rrf'
    Returns a list of dicts: [{document_id, score, rank, text, tags}, ...]
    """
    retriever = get_retriever(model)
    return retriever.search(query, top_k=top_k)
