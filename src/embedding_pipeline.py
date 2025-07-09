import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# Load cleaned dataset
df = pd.read_csv('../data/filtered_complaints.csv')

# Initialize chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

# Chunk all narratives
chunks = []
metadata = []
for idx, row in df.iterrows():
    chunks_from_doc = text_splitter.split_text(row['Cleaned_Narrative'])
    for chunk in chunks_from_doc:
        chunks.append(chunk)
        metadata.append({
            'complaint_id': row.get('Complaint ID', idx),
            'product': row['Product'],
        })

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all chunks
embeddings = model.encode(chunks, show_progress_bar=True)

# Convert to FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings).astype("float32"))

# Save FAISS index
faiss.write_index(index, "vector_store/faiss_index.index")

# Save metadata
with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump({'chunks': chunks, 'metadata': metadata}, f)

print("✅ Embeddings and metadata saved in vector_store/")
