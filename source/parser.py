import os
import re
import requests
from bs4 import BeautifulSoup

PAGES = [
    "Skeletron", "The_Twins", "The_Destroyer", "Duke_Fishron", "Empress_of_Light",
    "Terra_Blade", "Cell_Phone", "Rod_of_Discord", "Void_Bag", "Ankh_Charm",
    "Solar_Eclipse", "Pumpkin_Moon", "Dungeon", "The_Underworld", "Aether",
    "Guide", "Steampunker", "Dryad", "Housing", "NPC_happiness"
]

BASE_URL = "https://terraria.wiki.gg/wiki/"
OUTPUT_DIR = "dataset_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_page(page_name):
    url = BASE_URL + page_name
    response = requests.get(url, headers={"User-Agent": "RAG-Homework-Bot/1.0"})
    if response.status_code != 200:
        print(f"Failed to fetch {page_name}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"class": "mw-parser-output"})
    
    for tag in content_div.find_all(["script", "style", "table", "div"], class_=["navbox", "toc", "mw-editsection"]):
        tag.decompose()

    text = content_div.get_text(separator="\n")

    cleaned_text = re.sub(r'\n+', '\n', text).strip()

    file_path = os.path.join(OUTPUT_DIR, f"{page_name.replace('%27', '')}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Source URL: {url}\nTitle: {page_name}\n\n" + cleaned_text)
    print(f"Saved: {file_path}")

for p in PAGES:
    parse_page(p)