import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from logger import get_logger

logger = get_logger(__name__)

def web_search(search_terms, original_user_query, prompts, llm):
    def get_ddg_search_results(query, max_results=5):
        urls = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                for result in results:
                    if 'href' in result:
                        urls.append(result['href'])
        except:
            pass
        return urls
        

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
    final_searches = []
    urls = get_ddg_search_results(search_terms)
    if not urls:
        print("No valid URLs found.")
        return "Error: No valod URLs were found during search"
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
    results = web_search("HOI4 how to win playing as Germany", None, None, None)
    print(results)
