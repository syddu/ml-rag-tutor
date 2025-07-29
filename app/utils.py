from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/embeddings/sentencetransformers"
PATH_TO_CHUNKS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/chunks"

def retrieve_top_k_chunks_sentencetransformers(query: str, k = 1): 
    """
    Uses faiss to retrieve the top k most relevant text chunks based on a query and embeddings encoded with sentencetransformers
    """
    chunk_embeddings = np.array(json.load(open(f"{PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS}/contextualized_embeddings.json", "r", encoding="utf-8")), dtype='float32')
    vector_dimensions = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimensions)
    faiss.normalize_L2(chunk_embeddings)
    index.add(chunk_embeddings)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding], dtype='float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, k)
    chunks = json.load(open(f"{PATH_TO_CHUNKS}/contextualized_chunks.json", "r", encoding="utf-8"))
    res = [chunks[i] for i in indices[0]]
    
    return res
def retrieve_top_k_chunks_openai(query: str, k = 1):
    pass

if __name__ == "__main__":
    query = "Describe the gradient descent algorithm and the steps specifically"
    print(retrieve_top_k_chunks_sentencetransformers(query, k = 3))