"""
Поиск релевантных чанков через Qdrant.
"""

import sys

from sentence_transformers import SentenceTransformer

from qdrant_store import (
    COLLECTION_NAME,
    get_client,
)


QUERY_PREFIX = "query: "
MODEL_NAME = "intfloat/multilingual-e5-small"


def search(
    query: str,
    client,
    model: SentenceTransformer,
    top_k: int = 5,
):
    query_vector = model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32").tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        results.append(
            {
                "score": float(point.score),
                "source": payload.get("source"),
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
            }
        )

    return results


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Введите запрос: ").strip()

    if not query:
        raise SystemExit("Пустой запрос")

    client = get_client()
    model = SentenceTransformer(MODEL_NAME)

    results = search(
        query=query,
        client=client,
        model=model,
        top_k=5,
    )

    if not results:
        print("Ничего не найдено.")
        return

    for rank, result in enumerate(results, start=1):
        text = result["text"].replace("\n", " ").strip()

        print(
            f"\n[{rank}] "
            f"Score: {result['score']:.4f} | "
            f"Файл: {result['source']} | "
            f"Чанк #{result['chunk_id']}"
        )
        print(f"Текст: {text}...")


if __name__ == "__main__":
    main()