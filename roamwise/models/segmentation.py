"""
Segmentation models ("The Tools" -> Traveler Segmentation).

Traveler segmentation: KMeans over a synthetic user-survey corpus of six
preference dimensions, labeled with the nearest named archetype so the cluster
is interpretable instead of an opaque cluster id.

The proposal's second clustering job -- per-city KMeans over POI lat/lon into
walkable day-zones -- used to live here as `POIZoner`. Issue #72 moved day
assignment into the TOPTW model, which decides it jointly with selection and
ordering, so nothing called the zoner any more and issue #81 removed it. The
second independent KMeans the proposal describes is still in the project:
`pipeline/city_guide.py` clusters a city's POI coordinates to derive the
area-by-area structure of each city guide.
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

    def __init__(self, survey: pd.DataFrame = None):
        """`survey` overrides the shipped `user_survey.csv`, for one caller:
        `evaluation/survey_sensitivity.py` refits this on resampled and
        perturbed surveys to measure how much of the archetype assignment is
        a property of that file rather than of the traveler (#124). Nothing in
        the app passes it; the default path is unchanged."""
        df = pd.read_csv(DATA_DIR / "user_survey.csv") if survey is None else survey
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


if __name__ == "__main__":
    seg = TravelerSegmenter()
    example = {"budget": 0.2, "culture": 0.8, "nature": 0.3, "nightlife": 0.7, "relax": 0.2, "adventure": 0.6}
    result = seg.classify(example)
    print("Classified traveler:", result["archetype"])
    print("Ranked matches:", result["ranked_matches"][:3])
