This is a FastAPI-based backend that answers questions about MIT 6.3900: Introduction to Machine Learning using a RAG + OpenAI API pipeline.

It performs semantic search over the 6.3900 course textbook content (which I uploaded raw pdf files of) using Sentence Transformers and FAISS.

The system prompt is contained in the prompts folder, while the LLM call is in model.py. Will likely change up the prompts to see how performance is affected.

Currently set to using the top 20 chunks as context, though this number can be increased or decreased in model.py.