# RAG System for Culinary Advice

This project was developed as part of the [HSE NLP Course](http://wiki.cs.hse.ru/Глубинное_обучение_для_текстовых_данных_24/25) and applies Retrieval-Augmented Generation (RAG) techniques to build a Russian-language assistant for culinary advice.

The assistant can answer recipe-related questions and maintain context across multiple turns using query paraphrasing and history tracking.

---

## Key Features

- **LSH-based Vector Search**  
  Fast cosine similarity search over dense vectors using a custom Locality-Sensitive Hashing (LSH) implementation.

- **Hierarchical Chunk Splitting**  
  Custom recursive text splitter to divide long recipes into overlapping chunks, inspired by [RecursiveCharacterTextSplitter](https://langchain-doc.readthedocs.io/en/latest/modules/indexes/examples/textsplitter.html)

- **Title-Enhanced Embeddings**  
  Final chunk embeddings combine title and body vectors with configurable weights, improving search relevance.

- **Query Paraphrasing**  
  A paraphrasing model reformulates user follow-up queries based on interaction history.

- **Fallback Handling**  
  The assistant declines to answer if no relevant recipes are found with high similarity.

- **Multi-turn Dialogue Support**  
  History tracking and paraphrasing model enable contextual reformulation of user questions while keeping the generation model prompt short.

---

## Repository Structure

- `vector_db.py` — vector database with cosine similarity search via LSH
- `splitter.py` — recursive hierarchial text splitter
- `db_creation.py` — preprocessing, embedding generation, and database population
- `RAGModel.py` — RAG prompt construction, recipe retrieval, and response generation
- `history.py` — chat history tracking and paraphrasing prompt creation
- `history_RAG.py` — wrapper class combining RAG model with query reformulation and session memory
- `system_prompt.txt` — defines the expected behavior of the generation model
- `paraphrase_prompt.txt` — guides the paraphrasing model in rewording context-based questions
- `usage_examples.py` — examples of real queries and generated answers from the assistant


---

## References
  
[Original course repository](https://github.com/ashaba1in/hse-nlp/tree/main/2024)
