import asyncio
import aiohttp
from typing import List, Dict, Any
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from core.base_agent import BaseAgent, AgentState


class WebSearchAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a web search specialist agent. Your role is to:
1. Analyze search queries and optimize them for better results
2. Search the web for relevant information
3. Extract and summarize key information from search results
4. Provide accurate, up-to-date information to help with coding tasks

Focus on finding technical documentation, code examples, best practices, and troubleshooting information.
Always cite your sources and indicate the recency of information when possible."""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(state, "assistant", "No search query provided")

        search_query = state.current_task

        try:
            search_results = await self._perform_search(search_query)

            if not search_results:
                return self.add_message(
                    state,
                    "assistant",
                    f"No search results found for query: {search_query}",
                )

            summary = await self._summarize_results(search_results, search_query)

            return self.add_message(
                state,
                "assistant",
                summary,
                metadata={"search_results": search_results, "query": search_query},
            )

        except Exception as e:
            return self.add_message(state, "assistant", f"Search failed: {str(e)}")

    async def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        duckduckgo_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    duckduckgo_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; WebSearchBot/1.0)"
                    },
                ) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    return self._parse_duckduckgo_results(html)

            except Exception as e:
                print(f"Search error: {e}")
                return []

    def _parse_duckduckgo_results(self, html: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        results = []

        result_items = soup.find_all("div", class_="result")[
            : self.config.search.max_results
        ]

        for item in result_items:
            try:
                title_elem = item.find("a", class_="result__a")
                snippet_elem = item.find("a", class_="result__snippet")

                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get("href", "")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append({"title": title, "url": url, "snippet": snippet})

            except Exception as e:
                continue

        return results

    async def _summarize_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> str:
        results_text = "\n\n".join(
            [
                f"Title: {result['title']}\nURL: {result['url']}\nSummary: {result['snippet']}"
                for result in results
            ]
        )

        prompt = f"""Based on the search query "{query}", analyze and summarize these search results:

{results_text}

Provide a comprehensive summary that:
1. Directly addresses the search query
2. Highlights the most relevant information
3. Includes specific technical details when available
4. Notes any conflicting information
5. Suggests follow-up searches if needed

Focus on information that would be helpful for software development tasks."""

        return await self.generate_response(prompt)
