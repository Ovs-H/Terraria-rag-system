# Terraria RAG-System

Учебный проект по созданию RAG-системы для поиска знаний о Terraria. Проект собирает и структурирует информацию из тематических текстов, разбивает их на чанки, вычисляет эмбеддинги и позволяет искать релевантные фрагменты по пользовательским запросам.

Авторы: Bobrovskij K., Epifanov V., Bozoyan O.
Группа: 8I32

---

## Что умеет проект

- Парсит исходные данные из `dataset_raw` и/или внешних источников.
- Поддерживает чтение текстовых файлов `.txt`, `.md`, `.pdf`.
- Разбивает текст на чанки по словам с перекрытием.
- Считает эмбеддинги через `sentence-transformers`.
- Использует векторный поиск для поиска релевант��ых чанков.
- Работает как CLI-скрипт: индексирование и поиск выполняются отдельно.
- Поддерживает интеграцию с Qdrant вместо локального FAISS-индекса.

---

## Стек

- Python 3.10+
- `requests`, `beautifulsoup4` — парсинг/сбор данных
- `pypdf` — чтение PDF
- `sentence-transformers` — эмбеддинги
- `intfloat/multilingual-e5-small` — основная модель для embeddings
- `faiss-cpu` — локальный векторный индекс (текущая базовая версия)
- `qdrant-client` — работа с Qdrant
- `numpy`, `tqdm` — вспомогательные библиотеки
- Docker / Docker Compose — запуск Qdrant

> В текущей версии репозитория реализован базовый FAISS-подход. Однако проект уже подготовлен к переходу на Qdrant: это более удобный вариант для хранения векторов, метаданных и масштабирования.

---

## Архитектура

```text
dataset_raw/
    ↓
source/parser.py
    ↓
chunking + preprocessing
    ↓
SentenceTransformer (E5)
    ↓
FAISS / Qdrant
    ↓
source/search.py
    ↓
поиск релевантных чанков по запросу
```

Основной рабочий поток:

1. Данные загружаются в `dataset_raw`.
2. `source/parser.py` собирает/подготавливает текст.
3. `source/indexer.py` разбивает данные на чанки и вычисляет эмбеддинги.
4. Результаты сохраняются в векторный индекс (`faiss.index` либо Qdrant collection).
5. `source/search.py` отправляет запрос в модель и возвращает ближайшие чанки.

---

## Структура проекта

```text
.
├── README.md
├── .gitignore
├── requirments.txt
├── dataset_raw/
├── source/
│   ├── parser.py
│   ├── indexer.py
│   └── search.py
├── Presentations & Docs/
└── qdrant_storage/   # создаётся после запуска Qdrant через Docker
```

> Важно: файл с зависимостями называется `requirments.txt` (с опечаткой в имени). При установке используйте именно его.

---

## Быстрый старт

### 1. Клонируйте проект

```bash
git clone https://github.com/Ovs-H/Terraria-rag-system.git
cd Terraria-rag-system
```

### 2. Создайте виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Установите зависимости

```bash
pip install -r requirments.txt
```

Если вы планируете использовать Qdrant, также установите клиент:

```bash
pip install qdrant-client
```

### 4. Подготовьте данные

Положите исходные файлы в `dataset_raw/`.
Поддерживаются:

- `.txt`
- `.md`
- `.pdf`

### 5. Индексация

Запуск индексирования через FAISS:

```bash
python source/indexer.py --dataset dataset_raw --out index_store
```

Запуск индексирования в Qdrant требует предварительного запуска сервера Qdrant.

---

## Qdrant: как подключить

### 1. Запуск Qdrant через Docker

Создайте `docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: terraria-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Запуск:

```bash
docker compose up -d
```

После запуска Qdrant будет доступен по адресу:

- API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`

### 2. Вариант реализации

Идея:

- оставить `chunk_text()` и `SentenceTransformer()` без изменения;
- заменить сохранение в FAISS на сохранение в Qdrant collection;
- хранить в payload:
  - `text`
  - `source`
  - `chunk_id`
- в поиске использовать `query_points()` с вектором запроса.

Для этого обычно добавляют отдельный модуль вроде `source/qdrant_store.py`, который:

- подключается к Qdrant;
- проверяет наличие коллекции;
- создает её с правильной размерностью векторов;
- делает `upsert` чанков и эмбеддингов.

Это позволяет хранить метаданные рядом с вектором и удобно обновлять индекс после изменений в данных.

---

## Поиск по запросу

Пример поиска через текущую реализацию:

```bash
python source/search.py "Как получить Ankh Shield?"
```

Если индекс уже создан, скрипт загрузит его и вернёт top-k наиболее похожих чанков.

---

## Пример взаимодействия

```bash
python source/indexer.py --dataset dataset_raw --out index_store
python source/search.py "Какие предметы дают защиту от дебаффов?"
```

На выходе появляются релевантные фрагменты текста с указанием:

- `score`
- `source` (файл)
- `chunk_id`
- текст чанка

---

## TODO / Roadmap

- Перевести хранение векторов из FAISS в Qdrant
- Добавить API для обработки вопросов через HTTP
- Подключить LLM (например, Mistral AI) для генерации финального ответа на основе найденных чанков
- Добавить фильтрацию по файлам/темам/категориям
- Ул��чшить качество чанкинга и ранжирования
- Добавить тесты и CI

---

## Примечание

Этот проект служит учебным примером RAG-подхода: он демонстрирует базовую цепочку

`данные → чанки → эмбеддинги → поиск → релевантные фрагменты`,

и может быть расширен до полноценной вопросно-ответной системы с LLM и векторной БД.

---

## Полезные ссылки

- Qdrant: https://qdrant.tech/
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://faiss.ai/
- Terraria Wiki: https://terraria.wiki.gg/
