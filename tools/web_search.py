import re
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from logger import get_logger

logger = get_logger(__name__)

def web_search(search_terms, original_user_query, prompts, llm):
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
        
    def summarize_findings(search_results):
        values = {
            "query":original_user_query,
            "search_terms":search_terms,
            "search_results":search_results,
        }
        final_prompt = prompts["web_search_summarize_findings"].format(**values)
        return llm.invoke(final_prompt).content
    
    logger.info("Called Tool: Web Search")
    search_tool = DuckDuckGoSearchResults(num_results=5)
    final_searches = []
    results = search_tool.run(search_terms)
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
    formatted = "\n\n".join([f'Search {i + 1}: "{text}"' for i, text in enumerate(final_searches)])
    if not formatted:
        return prompts["web_search_tool_error"]
    logger.info("Summarizing web search findigns and returning...")
    return summarize_findings(formatted)

if __name__ == "__main__":
    results = web_search("HOI4 how to win playing as Germany")
    print(results)
