import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from dotenv import load_dotenv
import numpy as np
from contextualize_chunks import contextualize_all_chunks

PATH_TO_RAW_DATA = "/Users/sydneydu/Projects/intro_ml_rag_assistant/raw_data"
PATH_TO_TEXT_DATA = "/Users/sydneydu/Projects/intro_ml_rag_assistant/data"
PATH_TO_CHUNKS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/chunks"
PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/embeddings/sentencetransformers"
PATH_TO_OPENAI_EMBEDDINGS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/embeddings/openai"

def convert_raw_data_to_txt(path_to_raw_data = PATH_TO_RAW_DATA, path_to_text_data = PATH_TO_TEXT_DATA):
    """
    Takes pdf data stored in path_to_raw_data and saves it to a txt file in path_to_text_data
    """
    text=[]
    for chapter in sorted(os.listdir(path_to_raw_data)):
        if not chapter.lower().endswith(".pdf"):
            continue
        current_chapter = []
        doc = fitz.open(f"{path_to_raw_data}/{chapter}")
        for page in doc:
            current_chapter.append(page.get_text())  
        text.append(" ".join(current_chapter).replace("\xa0", " ").strip()) 
    for idx, chapter_text in enumerate(text):
        chapter_number = idx + 1
        with open(f"{path_to_text_data}/notes_by_chapter/chapter_{chapter_number}.txt", 'w', encoding='utf-8') as f:
            f.write(chapter_text)
    with open(f"{path_to_text_data}/notes.txt", 'w', encoding='utf-8') as f:
        f.write(" ".join(text).replace("\xa0", " ").strip())
    return text

def chunk_text_data_by_chapter(text = None, path_to_text_data = PATH_TO_TEXT_DATA, path_to_chunks = PATH_TO_CHUNKS):
    """
    Uses LangChain RecursiveCharacterTextSplitter to chunk the text data in path_to_text_data into semantically significant chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    if text is None:
        text = [open(f"{path_to_text_data}/notes_by_chapter/chapter_{chapter_number}.txt").read() for chapter_number in range(1,16)]
    chunks_by_chapter = []
    for doc in text:
        chunks_by_chapter.append([])
        chunks_by_chapter[-1].extend(text_splitter.split_text(doc))
    for idx, chunks in enumerate(chunks_by_chapter):
        chapter_number = idx + 1
        with open(f"{path_to_chunks}/chunks_by_chapter/chapter_{chapter_number}_chunks.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(chunks))
    return chunks_by_chapter

def embed_chunks_sentencetransformers(contextualized_chunks = None, model_name = "all-MiniLM-L6-v2"):
    if contextualized_chunks is None:
        contextualized_chunks = json.load(open(f"{PATH_TO_CHUNKS}/contextualized_chunks.json", "r", encoding="utf-8"))
    model = SentenceTransformer(model_name)
    embeddings = model.encode(contextualized_chunks)
    embeddings = [emb.tolist() for emb in embeddings]
    with open(f"{PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS}/contextualized_embeddings.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(embeddings))
    return embeddings

def embed_chunks_openai(contextualized_chunks = None, model_name = "text-embedding-3-small"):
    if contextualized_chunks is None:
        contextualized_chunks = json.load(open(f"{PATH_TO_CHUNKS}/contextualized_chunks.json", "r", encoding="utf-8"))
    load_dotenv()
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=model_name,
        input=contextualized_chunks,
        encoding_format="float"
    )
    embeddings = [item.embedding for item in response.data]
    with open(f"{PATH_TO_OPENAI_EMBEDDINGS}/contextualized_embeddings.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(embeddings))
    return embeddings

def tfidf_vectorizer(contextualized_chunks = None):
    vectorizer = TfidfVectorizer()
    if contextualized_chunks is None:
        contextualized_chunks = json.load(open(f"{PATH_TO_CHUNKS}/contextualized_chunks.json", "r", encoding = "utf-8"))
    tfidf_matrix = vectorizer.fit_transform(contextualized_chunks)
    return tfidf_matrix
    
def prepare_data():
    chunk_text_data_by_chapter(convert_raw_data_to_txt())
    # contextualized_chunks = contextualize_all_chunks() #Takes VERY long time to contextualize all, only do it if the whole text changes drastically
    embed_chunks_openai()
    tfidf_vectorizer()

if __name__ == "__main__":
    #embed more chunks as needed
    # prepare_data()
    print(tfidf_vectorizer())
    pass