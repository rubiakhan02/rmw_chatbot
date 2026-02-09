import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

BASE = "https://ritzmediaworld.com"
visited = set()
collected_text = []


# -------- GET INTERNAL LINKS --------
def get_links(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(BASE, a["href"])
            if urlparse(href).netloc == urlparse(BASE).netloc:
                links.append(href)

        return links
    except:
        return []


# -------- CLEAN TEXT EXTRACTION --------
def extract_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove junk
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        return " ".join(text.split())

    except:
        return ""


# -------- MAIN SCRAPER --------
seed_pages = [
    BASE,
    BASE + "/services",
    BASE + "/about",
]

for seed in seed_pages:
    print("Scanning:", seed)
    links = get_links(seed)

    for link in links:

        # Only target useful pages
        if any(k in link.lower() for k in [
            "service",
            "marketing",
            "advertising",
            "development",
            "about"
        ]):

            if link not in visited:
                visited.add(link)

                print("Scraping:", link)
                text = extract_text(link)

                if len(text) > 200:
                    collected_text.append(f"\n===== {link} =====\n")
                    collected_text.append(text)

                time.sleep(1)


# -------- SAVE OUTPUT --------
with open("website_data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(collected_text))

print("\n✅ Done — High quality data saved!")
