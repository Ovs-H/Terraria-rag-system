import os
import uuid
from typing import Iterable

from qdrant_client import QdrantClient, models


COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "terraria_chunks",
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_client() -> QdrantClient:
    kwargs = {
        "url": QDRANT_URL,
    }

    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY

    return QdrantClient(**kwargs)


def ensure_collection(
    client: QdrantClient,
    vector_size: int,
) -> None:
    if client.collection_exists(COLLECTION_NAME):
        collection = client.get_collection(COLLECTION_NAME)

        existing_size = collection.config.params.vectors.size

        if existing_size != vector_size:
            raise RuntimeError(
                f"Размерность существующей коллекции: {existing_size}, "
                f"размерность модели: {vector_size}"
            )

        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def make_point_id(source: str, chunk_id: int) -> str:
    """
    Один и тот же файл и номер чанка всегда получают одинаковый UUID.
    Благодаря этому повторная индексация обновляет точку,
    а не создаёт дубликат.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_id}"))


def upload_records(
    client: QdrantClient,
    records: list[dict],
    embeddings,
    batch_size: int = 64,
) -> None:
    vector_size = int(embeddings.shape[1])
    ensure_collection(client, vector_size)

    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        batch_embeddings = embeddings[start:start + batch_size]

        points = []

        for record, embedding in zip(batch_records, batch_embeddings):
            point_id = make_point_id(
                record["source"],
                record["chunk_id"],
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": record["text"],
                        "source": record["source"],
                        "chunk_id": record["chunk_id"],
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(
            f"[+] Загружено в Qdrant: "
            f"{min(start + batch_size, len(records))}/{len(records)}"
        )