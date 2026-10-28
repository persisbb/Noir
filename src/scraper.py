import os
import requests
from bs4 import BeautifulSoup


def scrape_page(url: str) -> str:
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()

    text = soup.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    url = "https://huggingface.co/docs"

    text = scrape_page(url)

    os.makedirs("data/raw", exist_ok=True)

    with open("data/raw/huggingface_docs.txt", "w", encoding="utf-8") as file:
        file.write(text)

    print("Scraping complete!")
    print(f"Characters collected: {len(text)}")
