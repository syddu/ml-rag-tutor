from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import numpy as np

PATH_TO_RAW_DATA = "/Users/sydneydu/Projects/intro_ml_rag_assistant/raw_data"
PATH_TO_TEXT_DATA = "/Users/sydneydu/Projects/intro_ml_rag_assistant/data"
PATH_TO_CHUNKS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/chunks"
PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS = (
    "/Users/sydneydu/Projects/intro_ml_rag_assistant/embeddings/sentencetransformers"
)
PATH_TO_OPENAI_EMBEDDINGS = (
    "/Users/sydneydu/Projects/intro_ml_rag_assistant/embeddings/openai"
)


def convert_raw_data_to_txt(
    path_to_raw_data=PATH_TO_RAW_DATA, path_to_text_data=PATH_TO_TEXT_DATA
):
    text = []
    for _, chapter in enumerate(os.listdir(path_to_raw_data)):
        reader = PdfReader(f"{path_to_raw_data}/{chapter}")
        for page in reader.pages:
            text.append(page.extract_text())
    text = " ".join(text).replace("\xa0", " ").strip()
    with open(f"{path_to_text_data}/notes.txt", "w", encoding="utf-8") as f:
        f.write(text)


def chunk_text_data(path_to_text_data=PATH_TO_TEXT_DATA, path_to_chunks=PATH_TO_CHUNKS):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for filename in os.listdir(path_to_text_data):
        with open(os.path.join(path_to_text_data, filename)) as f:
            doc = f.read()
        chunks.extend(text_splitter.split_text(doc))
    with open(f"{path_to_chunks}/chunks.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(chunks))
    return chunks


def embed_chunks_sentencetransformers(chunks=None, model_name="all-MiniLM-L6-v2"):
    if chunks is None:
        chunks = json.load(open(f"{PATH_TO_CHUNKS}/chunks.json", "r", encoding="utf-8"))
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    embeddings = [emb.tolist() for emb in embeddings]
    with open(
        f"{PATH_TO_SENTENCETRANSFORMERS_EMBEDDINGS}/embeddings.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(embeddings))
    return embeddings


def embed_chunks_openai(chunks=None, model_name="text-embedding-3-small"):
    if chunks is None:
        chunks = json.load(open(f"{PATH_TO_CHUNKS}/chunks.json", "r", encoding="utf-8"))
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=model_name, input=chunks, encoding_format="float"
    )
    embeddings = [item.embedding for item in response.data]
    with open(
        f"{PATH_TO_OPENAI_EMBEDDINGS}/embeddings.json", "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps(embeddings))
    return embeddings


if __name__ == "__main__":
    # embed more chunks as needed

    pass
