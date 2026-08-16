"""Builds the unstructured document corpus shared by semantic + keyword search:
city guide paragraphs and per-POI description snippets. Graph-RAG does not use
this corpus -- it traverses the structured knowledge graph directly.
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"


def load_documents() -> list[dict]:
    docs = []

    destinations = pd.read_csv(DATA_DIR / "destinations.csv").set_index("destination_id")
    for guide_path in sorted((DATA_DIR / "city_guides").glob("*.txt")):
        dest_id = guide_path.stem
        city = destinations.loc[dest_id, "city"]
        docs.append({
            "doc_id": f"guide::{dest_id}",
            "type": "city_guide",
            "destination_id": dest_id,
            "city": city,
            "text": guide_path.read_text(),
        })

    pois = pd.read_csv(DATA_DIR / "poi.csv")
    for _, p in pois.iterrows():
        city = destinations.loc[p.destination_id, "city"]
        text = f"{p['name']} ({p.category}) in {city}: {p.description}"
        docs.append({
            "doc_id": f"poi::{p.poi_id}",
            "type": "poi",
            "destination_id": p.destination_id,
            "city": city,
            "poi_id": p.poi_id,
            "name": p["name"],
            "text": text,
        })
    return docs


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


if __name__ == "__main__":
    docs = load_documents()
    print(f"{len(docs)} documents ({sum(d['type']=='city_guide' for d in docs)} guides, "
          f"{sum(d['type']=='poi' for d in docs)} POI snippets)")
