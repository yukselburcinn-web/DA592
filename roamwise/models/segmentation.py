"""
Segmentation models ("The Tools" -> Traveler & POI Segmentation).

Two distinct clustering jobs, as specified in the proposal:
  1. Traveler segmentation: KMeans over a synthetic user-survey corpus of six
     preference dimensions, labeled with the nearest named archetype so the
     cluster is interpretable instead of an opaque cluster id.
  2. POI geographic zoning: per-city KMeans over POI lat/lon to group sights
     into walkable daily zones, which the RouterAgent then sequences.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"

FEATURES = ["budget", "culture", "nature", "nightlife", "relax", "adventure"]


class TravelerSegmenter:
    """KMeans over the archetype survey data; maps raw user sliders to the
    closest archetype cluster, which the GraphIndex uses to fetch
    ArchetypeProfile -[PREFERS]-> POI edges."""

    def __init__(self):
        df = pd.read_csv(DATA_DIR / "user_survey.csv")
        self.df = df
        self.k = df.archetype.nunique()
        self.model = KMeans(n_clusters=self.k, n_init=10, random_state=42)
        self.model.fit(df[FEATURES])
        # map each fitted cluster to the majority archetype label within it
        df = df.copy()
        df["cluster"] = self.model.labels_
        self.cluster_to_archetype = (
            df.groupby("cluster")["archetype"]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )

    def classify(self, preferences: dict) -> dict:
        """preferences: dict with keys in FEATURES, values 0-1.
        Returns the matched archetype plus distances to every archetype
        centroid so the caller can show a confidence / runner-up."""
        x = pd.DataFrame([[preferences.get(f, 0.5) for f in FEATURES]], columns=FEATURES)
        cluster = int(self.model.predict(x)[0])
        archetype = self.cluster_to_archetype[cluster]

        centroids = self.model.cluster_centers_
        dists = np.linalg.norm(centroids - x.values, axis=1)
        ranked = sorted(
            zip((self.cluster_to_archetype[c] for c in range(self.k)), dists),
            key=lambda t: t[1],
        )
        return {
            "archetype": archetype,
            "cluster_id": cluster,
            "ranked_matches": [{"archetype": a, "distance": float(d)} for a, d in ranked],
        }


class POIZoner:
    """Per-city KMeans over POI coordinates -> geographic day-zones."""

    def zone(self, pois: list[dict], n_zones: int) -> dict[int, list[dict]]:
        if len(pois) <= n_zones:
            return {i: [p] for i, p in enumerate(pois)}
        coords = np.array([[p["lat"], p["lon"]] for p in pois])
        model = KMeans(n_clusters=n_zones, n_init=10, random_state=42)
        labels = model.fit_predict(coords)
        zones: dict[int, list[dict]] = {i: [] for i in range(n_zones)}
        for poi, label in zip(pois, labels):
            zones[int(label)].append(poi)
        return zones


if __name__ == "__main__":
    seg = TravelerSegmenter()
    example = {"budget": 0.2, "culture": 0.8, "nature": 0.3, "nightlife": 0.7, "relax": 0.2, "adventure": 0.6}
    result = seg.classify(example)
    print("Classified traveler:", result["archetype"])
    print("Ranked matches:", result["ranked_matches"][:3])

    from knowledge_graph.build_graph import GraphIndex  # noqa: E402
    idx = GraphIndex()
    pois = idx.city_pois("PAR")
    zoner = POIZoner()
    zones = zoner.zone(pois, n_zones=3)
    for z, members in zones.items():
        print(f"Zone {z}: {[m['name'] for m in members]}")
