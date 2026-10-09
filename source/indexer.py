"""
Читает файлы из dataset_raw/, режет на чанки (по словам),
считает эмбеддинги (с префиксом passage: для E5) и сохраняет FAISS-индекс.
"""

import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_store import get_client, upload_records

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}

def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf не установлен — не могу читать PDF")
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return ""

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Нарезает текст на чанки по СЛОВАМ, чтобы не разрывать слова пополам.
    chunk_size и overlap указываются в количестве слов.
    """
    words = text.split()
    if not words:
        return []
    
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap должен быть меньше chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
        
    return chunks

def collect_chunks(dataset_dir: Path, chunk_size: int, overlap: int):
    records = []
    files = [p for p in sorted(dataset_dir.rglob("*"))
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    for path in tqdm(files, desc="Читаю файлы"):
        try:
            text = read_file(path)
        except Exception as e:
            print(f"[!] Ошибка чтения {path}: {e}")
            continue
            
        for i, ch in enumerate(chunk_text(text, chunk_size, overlap)):
            records.append({
                "source": str(path.relative_to(dataset_dir)),
                "chunk_id": i,
                "text": ch,
            })
    return records

def main():
    parser = argparse.ArgumentParser(description="Chunking + indexing")
    parser.add_argument("--dataset", default="dataset_raw")
    parser.add_argument("--out", default="index_store")
    # 250 слов
    parser.add_argument("--chunk-size", type=int, default=250) 
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="Модель sentence-transformers (для E5 нужны префиксы)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise SystemExit(f"Папка {dataset_dir} не найдена")

    print(f"[+] Читаю {dataset_dir}")
    records = collect_chunks(dataset_dir, args.chunk_size, args.overlap)
    if not records:
        raise SystemExit("Не нашёл ни одного чанка. Проверь содержимое dataset_raw")
    print(f"[+] Чанков: {len(records)}")

    print(f"[+] Загружаю модель: {args.model}")
    model = SentenceTransformer(args.model)

    texts_with_prefix = [f"passage: {r['text']}" for r in records]
    
    print("[+] Считаю эмбеддинги...")
    embeddings = model.encode(
        texts_with_prefix,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True, 
    ).astype("float32")

    client = get_client()

    print("[+] Загружаю эмбеддинги в Qdrant...")
    upload_records(
        client=client,
        records=records,
        embeddings=embeddings,
        batch_size=args.batch_size,
    )
    
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "chunk_size_words": args.chunk_size,
            "overlap_words": args.overlap,
            "count": len(records),
        }, f, ensure_ascii=False, indent=2)

    print(f"[+] Готово. Индекс сохранён в {out_dir}/")

if __name__ == "__main__":
    main()