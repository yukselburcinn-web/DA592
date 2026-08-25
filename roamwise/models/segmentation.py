"""
Segmentation models ("The Tools" -> Traveler & POI Segmentation).

Two distinct clustering jobs, as specified in the proposal:
  1. Traveler segmentation: KMeans over a synthetic user-survey corpus of six
     preference dimensions, labeled with the nearest named archetype so the
     cluster is interpretable instead of an opaque cluster id.
  2. POI geographic zoning: per-city KMeans over POI lat/lon to group sights
     into walkable daily zones, which the RouterAgent then sequences.
"""
import math
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
    """Per-city KMeans over POI coordinates -> geographic day-zones.

    Plain KMeans optimizes only for geographic compactness, which on a real
    city is badly unbalanced: the dense historic centre collapses into one
    huge cluster while outlying sights each get a cluster of their own. Since
    one zone == one day of the trip, that produced the symptom issue #19
    reports -- a 5-day plan where one day held six sights and three days held
    a single (often unreachable) one. `balanced=True` therefore keeps the
    KMeans centroids but reassigns POIs under a per-zone capacity cap, so
    every day starts from a comparable number of candidates.
    """

    def zone(self, pois: list[dict], n_zones: int, balanced: bool = True) -> dict[int, list[dict]]:
        if len(pois) <= n_zones:
            # Every requested zone comes back, even the ones no POI landed in.
            # This used to return one zone per POI, so it silently returned
            # *fewer days than the traveler asked for* whenever candidates ran
            # short -- a 5-day trip with 3 sightseeing POIs became a 3-day
            # itinerary -- and an empty pool returned no zones at all, which
            # `_rebalance_days` then crashed on with "min() iterable argument
            # is empty". An empty zone is a legitimate state: the meal and
            # evening passes fill days the sightseeing pool could not (#63).
            zones: dict[int, list[dict]] = {i: [] for i in range(n_zones)}
            for i, poi in enumerate(pois):
                zones[i].append(poi)
            return zones
        coords = np.array([[p["lat"], p["lon"]] for p in pois])
        model = KMeans(n_clusters=n_zones, n_init=10, random_state=42)
        labels = model.fit_predict(coords)
        if balanced:
            labels = self._balance(coords, model.cluster_centers_, n_zones)
        zones: dict[int, list[dict]] = {i: [] for i in range(n_zones)}
        for poi, label in zip(pois, labels):
            zones[int(label)].append(poi)
        return zones

    @staticmethod
    def _balance(coords: np.ndarray, centers: np.ndarray, n_zones: int) -> np.ndarray:
        """Capacity-constrained reassignment against fixed KMeans centroids.

        Every (poi, zone) pair is considered cheapest-first and taken while
        the POI is unassigned and the zone is under its cap, so a POI only
        loses its nearest zone to a POI that wants it more. The cap is
        ceil(n / n_zones), which is the smallest cap that can still hold
        every POI. A final repair pass hands any zone that still came out
        empty the POI nearest to it, since an empty zone is an empty day --
        exactly the outcome this exists to prevent."""
        n = len(coords)
        capacity = math.ceil(n / n_zones)
        # distances[i][z] = how far POI i is from zone z's centroid
        distances = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)

        order = sorted(
            ((distances[i][z], i, z) for i in range(n) for z in range(n_zones)),
            key=lambda t: t[0],
        )
        labels = np.full(n, -1, dtype=int)
        counts = [0] * n_zones
        for _, i, z in order:
            if labels[i] == -1 and counts[z] < capacity:
                labels[i] = z
                counts[z] += 1

        for z in range(n_zones):
            if counts[z] > 0:
                continue
            donors = [i for i in range(n) if counts[labels[i]] > 1]
            if not donors:
                break
            take = min(donors, key=lambda i: distances[i][z])
            counts[labels[take]] -= 1
            labels[take] = z
            counts[z] += 1
        return labels


if __name__ == "__main__":
    seg = TravelerSegmenter()
    example = {"budget": 0.2, "culture": 0.8, "nature": 0.3, "nightlife": 0.7, "relax": 0.2, "adventure": 0.6}
    result = seg.classify(example)
    print("Classified traveler:", result["archetype"])
    print("Ranked matches:", result["ranked_matches"][:3])

    from roamwise.knowledge_graph.build_graph import GraphIndex  # noqa: E402
    idx = GraphIndex()
    pois = idx.city_pois("PAR")
    zoner = POIZoner()
    zones = zoner.zone(pois, n_zones=3)
    for z, members in zones.items():
        print(f"Zone {z}: {[m['name'] for m in members]}")
