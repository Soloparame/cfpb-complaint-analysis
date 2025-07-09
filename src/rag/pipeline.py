import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load vector index and metadata
index = faiss.read_index("vector_store/faiss_index.index")
with open("vector_store/metadata.pkl", "rb") as f:
    store = pickle.load(f)

chunks = store['chunks']
metadata = store['metadata']

# Load embedding model and LLM
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llm = pipeline("text-generation", model="gpt2", max_length=300)

# Prompt template
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. 
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

def retrieve_chunks(query, k=5):
    query_vector = embedder.encode([query])
    _, I = index.search(np.array(query_vector).astype("float32"), k)
    results = [(chunks[i], metadata[i]) for i in I[0]]
    return results

def generate_answer(question):
    results = retrieve_chunks(question)
    context_text = "\n---\n".join([res[0] for res in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    response = llm(prompt)[0]['generated_text'].split("Answer:")[-1].strip()
    return response, results

# Example usage
if __name__ == "__main__":
    question = "What are common issues reported with Buy Now, Pay Later services?"
    answer, sources = generate_answer(question)

    print("Q:", question)
    print("A:", answer)
    print("\nTop Sources:\n")
    for text, meta in sources[:2]:
        print(f"- Product: {meta['product']}")
        print(f"  Text: {text[:300]}...\n")
