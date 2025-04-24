import requests
from bs4 import BeautifulSoup, Comment
from duckduckgo_search import DDGS
from logger import get_logger
from googlesearch import search

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
    
    def get_google_search_results(query, max_results=5):
        return list(search(query, num_results=max_results))

    def scrape_article(url):
        def is_visible(element):
            if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'noscript']:
                return False
            if isinstance(element, Comment):
                return False
            return True
    
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if response.status_code != 200:
                return f"Failed to retrieve content from {url}"

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove irrelevant tags
            for tag in soup(["script", "style", "header", "footer", "nav", "form", "noscript"]):
                tag.decompose()

            # Use visible text only
            texts = soup.find_all(string=True)
            visible_texts = filter(is_visible, texts)
            visible_lines = [t.strip() for t in visible_texts if t.strip()]

            # De-duplicate and filter short/irrelevant lines
            seen = set()
            filtered = []
            for line in visible_lines:
                if line not in seen and len(line) > 30:
                    seen.add(line)
                    filtered.append(line)

            return "\n".join(filtered[:30]) if filtered else "No readable content found."

        except Exception as e:
            return f"Error scraping {url}: {e}"
        

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
    try:
        urls = get_ddg_search_results(search_terms)
    except:
        urls = get_google_search_results(search_terms)
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
