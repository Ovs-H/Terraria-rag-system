# Terraria RAG-System

Учебный проект RAG-поиска по Wiki Terraria.  
Система собирает страницы Terraria Wiki, режет их на чанки, считает эмбеддинги и ищет наиболее релевантные фрагменты по запросу пользователя.

Авторы: Bobrovskij K., Epifanov V., Bozoyan O.  
Группа: 8I32

---

## Что умеет проект

- Парсит страницы Terraria Wiki через `requests` + `BeautifulSoup4`.
- Сохраняет текст в `dataset_raw/*.txt`.
- Читает файлы форматов `.txt`, `.md`, `.pdf`.
- Режет текст на чанки по словам с перекрытием.
- Считает эмбеддинги моделью `intfloat/multilingual-e5-small`.
- Строит FAISS-индекс для быстрого поиска.
- Ищет top-k релевантных чанков по запросу через CLI.

---

## Стек

- Python 3.10+
- `requests`, `beautifulsoup4` — парсинг Wiki.
- `pypdf` — чтение PDF.
- `sentence-transformers` — эмбеддинги.
- `intfloat/multilingual-e5-small` — multilingual-модель для E5.
- `faiss-cpu` — векторный индекс.
- `numpy`, `tqdm` — вспомогательные библиотеки.

> В презентации указаны Qdrant, Mistral AI и LangChain.  
> В текущей реализации используется FAISS, а генерации ответа через LLM пока нет — система возвращает только релевантные чанки.  
> Qdrant, Mistral AI и LangChain можно добавить как развитие проекта.

---

## Структура проекта

```text
.
├── requirements.txt
├── README.md
├── dataset_raw/
└── source/
    ├── parser.py 
    ├── indexer.py 
    └── search.py 