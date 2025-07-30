from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/embeddings/sentencetransformers"
PATH_TO_CHUNKS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/chunks"
PATH_TO_TFIDF_MATRIX = "/Users/sydneydu/Projects/intro_ml_rag_assistant/tfidf_matrix"

def retrieve_top_k_chunks_sentencetransformers(query: str, k = 20): 
    """
    Uses faiss to retrieve the top k most relevant text chunks based on a query, tfidf vector similarity, and embeddings encoded with sentencetransformers
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding], dtype='float32')
    chunk_embeddings = np.array(json.load(open(f"{PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS}/contextualized_embeddings.json", "r", encoding="utf-8")), dtype='float32')
    
    vector_dimensions = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimensions)
    faiss.normalize_L2(chunk_embeddings)
    index.add(chunk_embeddings)
    faiss.normalize_L2(query_embedding)
    
    contextualized_chunks = json.load(open(f"{PATH_TO_CHUNKS}/contextualized_chunks.json", "r", encoding = "utf-8"))
    k_prime = min(k * 3, chunk_embeddings.shape[0])
    distances, indices = index.search(query_embedding, k_prime)
    semantic_rankings = [contextualized_chunks[index] for index in indices[0]]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contextualized_chunks).toarray()
    query_tfidf_vector = vectorizer.transform([query])
    dot_products = cosine_similarity(tfidf_matrix,query_tfidf_vector).flatten()
    lexical_rankings = [contextualized_chunks[i] for i in np.argsort(dot_products)[::-1][:k_prime]]
    
    return reciprocal_rank_fusion(semantic_rankings, lexical_rankings, k)
    
def reciprocal_rank_fusion(semantic_rankings, lexical_rankings, k=20):
    """
    Fuses the results of semantic and lexical retrieval by ranking each chunk with RRF
    """
    overall_rankings = defaultdict(int)
    for rank, chunk in enumerate(semantic_rankings):
        overall_rankings[chunk] += 1 / (60 + rank + 1)
    for rank, chunk in enumerate(lexical_rankings):
        overall_rankings[chunk] += 1 / (60 + rank + 1)    
    top_k_chunks = []
    for chunk, score in sorted(overall_rankings.items(), key = lambda x: x[1], reverse=True)[:k]:
        top_k_chunks.append(chunk)
    print(top_k_chunks)
    return top_k_chunks

if __name__ == "__main__":
    query = "Describe the gradient descent algorithm and the steps specifically"
    print(retrieve_top_k_chunks_sentencetransformers(query, k = 3))