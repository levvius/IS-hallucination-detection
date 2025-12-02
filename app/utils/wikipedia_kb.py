import wikipedia
import re
from typing import List, Dict


# Seed topics for Knowledge Base
SEED_TOPICS = [
    # Science
    "Albert Einstein", "World War II", "Python (programming language)",
    "Artificial intelligence", "COVID-19", "Barack Obama",
    "Linux", "Microsoft", "Climate change", "Quantum mechanics",
    "New York City", "Mount Everest", "Tesla, Inc.", "Amazon (company)",
    "Google", "Neural network", "Machine learning", "Data science"
]


def build_kb_snippets(topics: List[str] = None, max_sentences_per_page: int = 15) -> List[Dict[str, str]]:
    """
    Build knowledge base snippets from Wikipedia pages.

    Args:
        topics: List of Wikipedia topics to fetch. If None, uses SEED_TOPICS.
        max_sentences_per_page: Maximum number of sentences to process per page.

    Returns:
        List of dicts with keys: title, snippet, source
    """
    if topics is None:
        topics = SEED_TOPICS

    docs = []

    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            text = page.content
            source_url = page.url
        except Exception as e:
            try:
                text = wikipedia.summary(topic, auto_suggest=True)
                source_url = f"wikipedia:{topic}"
            except Exception as e2:
                print(f"Warning: could not fetch {topic}: {e2}")
                continue

        # Split into sentences and group into snippets (~2-3 sentences each)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cur = []

        for s in sentences[:max_sentences_per_page * 3]:
            cur.append(s.strip())
            if len(cur) >= 3:
                snippet = " ".join(cur)
                docs.append({
                    "title": topic,
                    "snippet": snippet,
                    "source": source_url
                })
                cur = []

        # Add remaining sentences as final snippet
        if cur:
            docs.append({
                "title": topic,
                "snippet": " ".join(cur),
                "source": source_url
            })

    return docs
