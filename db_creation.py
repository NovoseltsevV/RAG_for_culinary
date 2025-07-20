import torch
from vector_db import LSHdatabase
from splitter import RecursiveSplitter
from functools import partial
from tqdm.auto import tqdm
import numpy as np


def split_dataset(example, index, splitter: RecursiveSplitter) -> dict:
    return {
        "id": index,
        "name": example["name"],
        "ingredients": example["ingredients"],
        "chunks": splitter.split_text(example["text"])
    }

def create_database(
        dataset,
        embedding_model,
        chunk_size: int, 
        chunk_overlap: int,
        alpha: float = 0.5
) -> LSHdatabase:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splitter = RecursiveSplitter(chunk_size, chunk_overlap)
    chunk_dataset = dataset.map(
        partial(split_dataset, splitter=splitter),
        with_indices=True
    )
    lsh_db = LSHdatabase(k=16, L=32, dim=1024)

    text_chunks_numbered = []
    for chunk_dict in chunk_dataset:
        id = chunk_dict["id"]
        name = chunk_dict["name"]
        ingredient = chunk_dict["ingredients"]
        values = chunk_dict["chunks"]
        for chunk in values:
            text_chunks_numbered.append((id, chunk, name, ingredient))

    ids, text_chunks, names, ingredients = zip(*text_chunks_numbered)
    title_vectors = embedding_model.encode(
        names, batch_size=512,
        convert_to_numpy=True,
        device=device, normalize_embeddings=True, 
        show_progress_bar=True
    )
    descr_vectors = embedding_model.encode(
        text_chunks, batch_size=512,
        convert_to_numpy=True,
        device=device, normalize_embeddings=True, 
        show_progress_bar=True
    )
    for i in tqdm(range(len(descr_vectors))):
        final_v = alpha * descr_vectors[i] + (1 - alpha) * title_vectors[i]
        final_v = final_v / np.linalg.norm(final_v)
        lsh_db.add_vector(final_v, i)
        info_dict = {
            "id": ids[i],
            "text": text_chunks[i],
            "name": names[i],
            "ingredients": ingredients[i]
        }
        lsh_db.add_info(info_dict, i)
    return lsh_db

def semantic_search(
        db: LSHdatabase,
        query: str,
        embedding_model,
        limit: int = 10
) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_vector = embedding_model.encode(
        query, normalize_embeddings=True,
        device=device, convert_to_numpy=True
    )
    return db.find_nearest(query_vector, limit)
