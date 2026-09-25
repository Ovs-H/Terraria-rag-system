"""
Загружает индекс и ищет релевантные чанки по запросу.
Использует префикс "query: " для моделей E5.
"""

import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# КРИТИЧЕСКИ ВАЖНО: Для моделей E5 добавляем префикс "query: " к запросам.
# (Если смените модель на не-E5, например bge-m3 или rubert, поставьте "")
QUERY_PREFIX = "query: "

def load(out_dir: str = "index_store"):
    out = Path(out_dir)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(out / "faiss.index"))
    records = [json.loads(line) for line in
               (out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    model = SentenceTransformer(cfg["model"])
    return index, records, model, cfg

def search(query: str, index, records, model, top_k: int = 5):
    # Добавляем префикс к запросу
    q = model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, idxs = index.search(q, top_k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        results.append((float(score), records[i]))
    return results

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Введите запрос: ").strip()
        
    if not query:
        raise SystemExit("Пустой запрос")

    index, records, model, cfg = load()
    print(f"\n[Модель: {cfg['model']} | Чанков: {cfg['count']} | Размерность: {cfg['dim']}]\n")
    print("-" * 60)

    results = search(query, index, records, model)
    
    if not results:
        print("Ничего не найдено.")
        return

    for rank, (score, r) in enumerate(results, 1):
        print(f"\n[{rank}] Score: {score:.4f} | Файл: {r['source']} | Чанк #{r['chunk_id']}")
        # Выводим превью текста, заменяя переносы строк на пробелы для красоты
        text_preview = r["text"].replace("\n", " ").strip()
        print(f"Текст: {text_preview}...")
        
    print("\n" + "-" * 60)

if __name__ == "__main__":
    main()