## Task 2: Chunking & Embedding Summary

To prepare narratives for semantic search, we implemented a chunking strategy using LangChain's `RecursiveCharacterTextSplitter`. We selected a `chunk_size` of 300 and a `chunk_overlap` of 50. This balance ensures each chunk maintains enough semantic coherence while allowing overlap to preserve context across chunk boundaries.

For embedding, we selected the `sentence-transformers/all-MiniLM-L6-v2` model. It offers a strong trade-off between speed and accuracy, making it ideal for large-scale document embedding tasks. With 384 dimensions, it is lightweight yet powerful enough to capture semantic similarity in consumer complaint narratives.

FAISS was chosen as the vector store due to its performance and wide adoption in production-scale search tasks. Each vector is stored alongside metadata such as `Complaint ID` and `Product` category, enabling traceability during retrieval.
