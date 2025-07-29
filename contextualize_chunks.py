from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import hashlib
load_dotenv()
TOPICS = {1: "Introduction",
          2: "Regression",
          3: "Gradient Descent",
          4: "Classification",
          5: "Feature Representation",
          6: "Neural Networks",
          7: "Convolutional Neural Networks",
          8: "Representation Learning (Autoencoders)",
          9: "Transformers",
          10: "Markov Decision Processes",
          11: "Reinforcement Learning",
          12: "Non-Parametric Models",
          13: "Appendix A: Matrix Derivative Common Cases",
          14: "Appendix B: Optimizing Neural Networks",
          15: "Appendix C: Supervised Learning in a Nutshell",
          }
CONTEXTUALIZE_PROMPT = """We are given Chapter #{chapter_number}: {chapter_topic} in the 6.3900 MIT Intro to Machine Learning textbook:
                        <document> {chapter_content} </document>
                        Here is the chunk we want to situate within the chapter and whole document:
                        <chunk> {chunk} </chunk>
                        Please give a short succinct context to situate this chunk within the chapter and overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct chunk context and nothing else."""
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def contextualize_chapter_chunks(chapter_number, chunks):
    """
    Returns a list of contextualized chunks for a single chapter
    """
    chapter_topic = TOPICS[chapter_number]
    with open(f"data/notes_by_chapter/chapter_{chapter_number}.txt", "r") as f:
        chapter_content = f.read()
    prompt_hash = hashlib.sha256(f"Chapter Number: {chapter_number} || Topic: {chapter_topic}".encode('utf-8')).hexdigest()
    contextualized_chunks = []
    for chunk in chunks:
        prompt = CONTEXTUALIZE_PROMPT.format(chapter_number=chapter_number, chapter_topic = chapter_topic, chapter_content=chapter_content, chunk = chunk)
        response = client.responses.create(model = "gpt-4o-mini", input = prompt, user = prompt_hash)
        contextualized_chunks.append(f"Context: {response.output[0].content[0].text}\nChunk: {chunk}")
    print(contextualized_chunks)
    return contextualized_chunks

def contextualize_all_chunks(chapters = None):
    """
    Returns a flattened list of all contextualized chunks across every chapter
    """
    if chapters is None:
        chapters = [json.load(open(f"chunks/chunks_by_chapter/chapter_{chapter_number}_chunks.json", "r")) for chapter_number in range(1,16)]
    contextualized_chunks = []
    for idx, chunks in enumerate(chapters):
        chapter_number = idx + 1
        incoming = contextualize_chapter_chunks(chapter_number, chunks)
        contextualized_chunks.extend(incoming)
    with open("chunks/contextualized_chunks2.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(contextualized_chunks))
    return contextualized_chunks

if __name__ == "__main__":
    contextualize_all_chunks()
