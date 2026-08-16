"""Fusion RAG Agent: orchestrates the three-component retrieval pipeline,
prioritizing Graph-RAG for multi-hop relational queries and falling back to
semantic/keyword search for descriptive content -- then grounds a short
narrative in exactly the retrieved snippets (so downstream text stays
attributable, reducing geographic hallucination)."""
from agents.llm_client import LLMClient, get_default_llm_client
from retrieval.fusion import FusionRetriever


class FusionRAGAgent:
    def __init__(self, retriever: FusionRetriever = None, llm: LLMClient = None):
        self.retriever = retriever or FusionRetriever()
        self.llm = llm or get_default_llm_client()

    def run(self, query: str, destination_id: str, archetype: str = None,
            config: str = "fusion", top_k: int = 8) -> dict:
        results = self.retriever.retrieve(
            query, config=config, destination_id=destination_id, archetype=archetype, top_k=top_k,
        )
        narrative = self._narrate(query, results)
        return {
            "query": query,
            "config": config,
            "results": results,
            "narrative": narrative,
        }

    def _narrate(self, query: str, results: list[dict]) -> str:
        if not results:
            return "No retrieval context available (standard-prompting configuration)."
        bullets = "\n".join(f"- {r['text']}" for r in results[:6])
        prompt = f"""
        Query: {query}
        Grounded context retrieved from the knowledge base:
        {bullets}
        """
        return self.llm.complete(
            system="Summarize only the facts present in the retrieved context. Do not invent places.",
            prompt=prompt,
        )


if __name__ == "__main__":
    agent = FusionRAGAgent()
    out = agent.run(
        "museums and landmarks near a transport hub for a culture-loving traveler",
        destination_id="ROM", archetype="Culture Enthusiast",
    )
    print(out["narrative"])
