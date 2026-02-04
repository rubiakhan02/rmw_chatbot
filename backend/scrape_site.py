import requests
from bs4 import BeautifulSoup

urls = [
    "https://ritzmediaworld.com/",
    "https://ritzmediaworld.com/about",
    "https://ritzmediaworld.com/services",
    "https://ritzmediaworld.com/contact"
]

all_text = ""

for url in urls:
    print("Scraping:", url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    all_text += text + "\n"

# Save scraped text
with open("website_data.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("Done. Data saved in website_data.txt")
