import re
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_mistralai import ChatMistralAI

def web_search(query, prompts, llm):
    def extract_urls(text):
        return re.findall(r'https?://[^\s,]+', text)

    def scrape_article(url):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if response.status_code != 200:
                return f"Failed to retrieve content from {url}"
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article") or soup.find("main")
            paragraphs = article.find_all("p") if article else soup.find_all("p")
            text_content = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
            return text_content if text_content else "No readable content found."
        except Exception as e:
            return f"Error fetching {url}: {e}"
        
    search_tool = DuckDuckGoSearchResults(num_results=5)
    final_searches = []
    results = search_tool.run(query)
    urls = extract_urls(results)
    if not urls:
        print("No valid URLs found.")
        return
    for url in urls:
        try:
            content = scrape_article(url)
            final_searches.append(content)
        except:
            pass
    return "\n\n".join(final_searches)

if __name__ == "__main__":
    results = web_search("HOI4 how to win playing as Germany")
    print(results)
