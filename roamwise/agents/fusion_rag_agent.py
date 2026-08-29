"""Fusion RAG Agent: orchestrates the three-component retrieval pipeline,
prioritizing Graph-RAG for multi-hop relational queries and falling back to
semantic/keyword search for descriptive content -- then grounds a short
narrative in exactly the retrieved snippets (so downstream text stays
attributable, reducing geographic hallucination)."""
import re

from roamwise.agents.llm_client import LLMClient, get_default_llm_client
from roamwise.retrieval.fusion import FusionRetriever

# Graph-retrieved doc text ends in a "[graph: reason]" attribution tag (see
# graph_search.py) so the Retrieved-context tab can explain why a POI
# surfaced. That tag is internal retrieval bookkeeping, not something a
# reader of the narrative needs -- keep it out of the text fed to the "LLM"
# synthesis step (the raw r['text'] with the tag is still what the
# Retrieved-context tab displays).
_SOURCE_TAG_RE = re.compile(r"\s*\[[^\[\]]*\]\s*$")


class FusionRAGAgent:
    def __init__(self, retriever: FusionRetriever = None, llm: LLMClient = None):
        self.retriever = retriever or FusionRetriever()
        self.llm = llm or get_default_llm_client()

    def run(self, query: str, destination_id: str, archetype: str = None,
            config: str = "fusion", top_k: int = 8, narrate: bool = True,
            arrival_hub_id: str = None) -> dict:
        """narrate=False skips the LLM paraphrase and returns only `facts`.

        The orchestrator passes narrate=False because it feeds this straight
        into another LLM prompt: paraphrasing structured facts with one
        generation just to hand the result to a second generation costs a
        full model pass and loses information twice over (issue #57).

        arrival_hub_id is passed through to retrieval, where the graph
        component uses it to anchor its traversal (issue #126). It reached the
        router already; retrieval was the half of the wire that was never
        pulled.
        """
        results = self.retriever.retrieve(
            query, config=config, destination_id=destination_id, archetype=archetype, top_k=top_k,
            arrival_hub_id=arrival_hub_id,
        )
        facts = self._facts(query, results)
        return {
            "query": query,
            "config": config,
            "results": results,
            "facts": facts,
            "narrative": self._narrate(facts) if narrate else None,
        }

    def _facts(self, query: str, results: list[dict]) -> str:
        """The grounded, deterministic summary of what retrieval returned.

        This is the honest input to any downstream reasoning step: every line
        is a retrieved snippet verbatim, so nothing here can be an invention.
        """
        if not results:
            return "No retrieval context available (standard-prompting configuration)."
        bullets = "\n".join(f"- {_SOURCE_TAG_RE.sub('', r['text'])}" for r in results[:6])
        # Joined rather than an indented triple-quoted f-string for the same
        # reason as orchestrator._synthesize: bullets is already flush-left,
        # so splicing it into an indented template defeats dedent and the
        # template lines render as a Markdown code block.
        return "\n".join([
            f"Query: {query}",
            "Grounded context retrieved from the knowledge base:",
            bullets,
        ])

    def _narrate(self, facts: str) -> str:
        return self.llm.complete(
            system="Summarize only the facts present in the retrieved context. Do not invent places.",
            prompt=facts,
        )


# Demo blocks below take their city from the catalogue rather than naming one:
# a hardcoded code prints nothing at all once that city stops shipping.
def _demo_city():
    import pandas as pd
    from pathlib import Path as _P
    d = _P(__file__).resolve().parents[1] / "data" / "destinations.csv"
    return pd.read_csv(d).destination_id.iloc[0]


if __name__ == "__main__":
    agent = FusionRAGAgent()
    out = agent.run(
        "museums and landmarks near a transport hub for a culture-loving traveler",
        destination_id=_demo_city(), archetype="Culture Enthusiast",
    )
    print(out["narrative"])
