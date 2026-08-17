"""
RoamWise synthetic data generator.

The original proposal calls for Kaggle / TripAdvisor / OpenStreetMap / Wikidata
sourced data. This sandbox has no credentialed access to those APIs, so this
script procedurally generates a *structurally realistic* stand-in dataset with
the same shape: destinations, POIs, transport hubs, monthly tourism-demand
time series (with trend + seasonality + a COVID-era shock, like real arrivals
data), and free-text city guides for the semantic/keyword retrieval layers.

Run once: `python generate_data.py`. Output lands in this directory as CSV /
JSON / txt files that every other module reads.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Destinations
# ---------------------------------------------------------------------------
DESTINATIONS = [
    # id, city, country, lat, lon, budget_level(1-3), tags
    ("IST", "Istanbul", "Turkey", 41.0082, 28.9784, 1, ["culture", "history", "food", "nightlife"]),
    ("PAR", "Paris", "France", 48.8566, 2.3522, 3, ["culture", "romance", "art", "shopping"]),
    ("ROM", "Rome", "Italy", 41.9028, 12.4964, 2, ["history", "culture", "food", "religion"]),
    ("BCN", "Barcelona", "Spain", 41.3874, 2.1686, 2, ["beach", "art", "nightlife", "culture"]),
    ("AMS", "Amsterdam", "Netherlands", 52.3676, 4.9041, 2, ["culture", "nightlife", "art", "nature"]),
    ("PRG", "Prague", "Czechia", 50.0755, 14.4378, 1, ["history", "budget", "nightlife", "culture"]),
    ("VIE", "Vienna", "Austria", 48.2082, 16.3738, 3, ["culture", "music", "history", "luxury"]),
    ("LIS", "Lisbon", "Portugal", 38.7223, -9.1393, 1, ["beach", "food", "budget", "culture"]),
]
destinations_df = pd.DataFrame(
    DESTINATIONS, columns=["destination_id", "city", "country", "lat", "lon", "budget_level", "tags"]
)
destinations_df["tags"] = destinations_df["tags"].apply(json.dumps)
destinations_df.to_csv(HERE / "destinations.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Points of interest (POIs) -- procedurally scattered around each city
# ---------------------------------------------------------------------------
POI_TEMPLATES = {
    "IST": [
        ("Hagia Sophia", "landmark", 4.9, 0, "A former Byzantine cathedral and Ottoman mosque, Hagia Sophia is a UNESCO World Heritage icon blending Christian and Islamic art under a vast dome."),
        ("Topkapi Palace", "museum", 4.7, 2, "The opulent former residence of Ottoman sultans, with treasury rooms, harem quarters and sweeping Bosphorus views."),
        ("Grand Bazaar", "shopping", 4.4, 1, "One of the world's oldest covered markets, a maze of over 4,000 shops selling carpets, ceramics, spices and jewelry."),
        ("Blue Mosque", "religion", 4.7, 0, "An active mosque famed for its cascading domes and hand-painted blue Iznik tiles."),
        ("Bosphorus Strait Cruise", "nature", 4.6, 2, "A scenic boat trip along the strait separating Europe and Asia, passing waterfront palaces and fortresses."),
        ("Karakoy Waterfront", "food", 4.3, 1, "A trendy district of fish restaurants, cafes and galleries along the Golden Horn."),
        ("Istiklal Street", "nightlife", 4.2, 1, "A pedestrian avenue packed with bars, live music venues and a historic tram running its length."),
        ("Basilica Cistern", "landmark", 4.6, 0, "An underground Byzantine reservoir with hundreds of columns, softly lit and eerily beautiful."),
        ("Suleymaniye Mosque", "religion", 4.6, 0, "A grand Ottoman imperial mosque designed by architect Mimar Sinan, with panoramic city views."),
        ("Princes' Islands", "nature", 4.5, 2, "Car-free islands in the Sea of Marmara reachable by ferry, known for pine forests and horse-drawn carriages."),
    ],
    "PAR": [
        ("Eiffel Tower", "landmark", 4.7, 2, "The iron lattice tower that has defined the Paris skyline since 1889, with observation decks over the city."),
        ("Louvre Museum", "museum", 4.8, 2, "The world's largest art museum, home to the Mona Lisa and the Venus de Milo inside a former royal palace."),
        ("Montmartre & Sacre-Coeur", "culture", 4.6, 1, "A hilltop artists' quarter with cobbled streets, cabarets and the white-domed Sacre-Coeur basilica."),
        ("Champs-Elysees", "shopping", 4.3, 3, "A grand boulevard of flagship boutiques and cafes running from the Arc de Triomphe."),
        ("Musee d'Orsay", "museum", 4.7, 1, "An Impressionist and Post-Impressionist collection housed in a former Belle Epoque railway station."),
        ("Le Marais", "nightlife", 4.4, 2, "A historic district of narrow lanes, art galleries, cocktail bars and falafel stands."),
        ("Notre-Dame Cathedral", "religion", 4.6, 0, "The Gothic cathedral on the Ile de la Cite, under restoration after the 2019 fire but open to visit."),
        ("Seine River Cruise", "nature", 4.5, 2, "An evening boat ride past illuminated monuments along the river that splits the city."),
        ("Luxembourg Gardens", "nature", 4.6, 0, "Formal French gardens with fountains, statues and a palace, popular with picnicking Parisians."),
        ("Palace of Versailles", "landmark", 4.7, 2, "The lavish former royal residence outside Paris, famed for its Hall of Mirrors and manicured gardens."),
    ],
    "ROM": [
        ("Colosseum", "landmark", 4.8, 2, "The largest ancient amphitheater ever built, once host to gladiatorial contests, now Rome's defining ruin."),
        ("Roman Forum", "history", 4.6, 1, "The sprawling ruins of the political and commercial heart of ancient Rome."),
        ("Vatican Museums & Sistine Chapel", "museum", 4.7, 2, "A vast papal art collection culminating in Michelangelo's frescoed Sistine Chapel ceiling."),
        ("St. Peter's Basilica", "religion", 4.8, 0, "The largest church in the world, centerpiece of Vatican City, topped by Michelangelo's dome."),
        ("Trevi Fountain", "landmark", 4.7, 0, "A Baroque fountain where tradition holds that a coin tossed over the shoulder ensures a return to Rome."),
        ("Trastevere", "nightlife", 4.5, 1, "A cobblestoned neighborhood of trattorias, wine bars and ivy-draped facades across the Tiber."),
        ("Pantheon", "landmark", 4.7, 0, "A remarkably preserved Roman temple with the world's largest unreinforced concrete dome."),
        ("Piazza Navona", "culture", 4.5, 0, "A Baroque square built over an ancient stadium, ringed by cafes and sculpted fountains."),
        ("Borghese Gallery & Gardens", "museum", 4.6, 1, "An intimate villa museum of Bernini sculpture set in a large landscaped park."),
        ("Campo de' Fiori Market", "food", 4.2, 1, "A lively morning produce market that turns into an aperitivo hub by evening."),
    ],
    "BCN": [
        ("Sagrada Familia", "landmark", 4.8, 2, "Gaudi's still-unfinished basilica, a riot of organic spires and stained glass, under construction since 1882."),
        ("Park Guell", "nature", 4.6, 1, "A whimsical Gaudi-designed park of mosaic terraces and gingerbread-style pavilions overlooking the city."),
        ("La Rambla", "shopping", 4.0, 1, "A tree-lined pedestrian boulevard running from Placa Catalunya to the old harbor."),
        ("Gothic Quarter", "culture", 4.6, 0, "A labyrinth of medieval lanes, Roman remains and Gothic cathedrals in the old city center."),
        ("Barceloneta Beach", "beach", 4.3, 0, "The city's main urban beach, backed by seafood restaurants and beach bars."),
        ("Casa Batllo", "landmark", 4.6, 2, "A Gaudi-renovated townhouse with a skeletal, dragon-scaled facade on Passeig de Gracia."),
        ("El Born nightlife district", "nightlife", 4.4, 1, "Narrow streets of cocktail bars, tapas spots and design boutiques near the Picasso Museum."),
        ("Picasso Museum", "museum", 4.5, 1, "An extensive collection tracing Picasso's early development, housed in five medieval palaces."),
        ("Camp Nou", "culture", 4.4, 2, "FC Barcelona's home stadium, one of the largest in Europe, with a club museum and tours."),
        ("Montjuic Hill", "nature", 4.5, 1, "A hilltop park with a castle, gardens and city views, reachable by cable car."),
    ],
    "AMS": [
        ("Anne Frank House", "museum", 4.7, 1, "The canal house where Anne Frank hid during WWII, now a moving museum on her diary and the Holocaust."),
        ("Rijksmuseum", "museum", 4.8, 2, "The national museum of art and history, home to Rembrandt's The Night Watch."),
        ("Van Gogh Museum", "museum", 4.7, 2, "The world's largest collection of Van Gogh paintings and letters."),
        ("Canal Ring Cruise", "nature", 4.6, 1, "A boat tour along the UNESCO-listed 17th-century canal belt lined with gabled houses."),
        ("Jordaan District", "culture", 4.5, 0, "A former working-class quarter now full of galleries, boutiques and cozy brown cafes."),
        ("Vondelpark", "nature", 4.6, 0, "The city's largest park, popular for picnics, cycling and open-air concerts in summer."),
        ("Red Light District", "nightlife", 4.0, 0, "A historic canal-side nightlife area known for its bars, coffee shops and neon-lit streets."),
        ("Albert Cuyp Market", "food", 4.3, 0, "The Netherlands' busiest street market, stalls of stroopwafels, cheese and fresh fish."),
        ("NEMO Science Museum", "culture", 4.3, 1, "A green-hulled harbor-front building of interactive science exhibits, family-friendly."),
        ("Heineken Experience", "culture", 4.2, 2, "An interactive tour of the former Heineken brewery with tastings."),
    ],
    "PRG": [
        ("Prague Castle", "landmark", 4.7, 1, "The largest ancient castle complex in the world, seat of Czech rulers for over a thousand years."),
        ("Charles Bridge", "landmark", 4.6, 0, "A Gothic stone bridge lined with baroque statues, crossing the Vltava at dawn is a local ritual."),
        ("Old Town Square", "culture", 4.6, 0, "A medieval square dominated by the Astronomical Clock and Tyn Church's twin spires."),
        ("Astronomical Clock", "landmark", 4.3, 0, "A 15th-century clock that puts on an hourly parade of mechanical apostle figures."),
        ("Jewish Quarter (Josefov)", "history", 4.5, 1, "A preserved ghetto of historic synagogues and the Old Jewish Cemetery."),
        ("Petrin Hill", "nature", 4.5, 0, "A green hill with a mini Eiffel-style lookout tower and rose gardens above the city."),
        ("Wenceslas Square", "shopping", 4.1, 1, "The commercial heart of the New Town, site of many pivotal moments in Czech history."),
        ("Vltava Beer Gardens", "nightlife", 4.4, 0, "Riverside beer gardens serving Czech pilsners with views of Prague Castle."),
        ("Lennon Wall", "culture", 4.0, 0, "An ever-changing wall of graffiti and lyrics that became a symbol of youthful dissent."),
        ("Municipal House", "culture", 4.4, 1, "An Art Nouveau concert hall and cafe, one of Prague's most ornate buildings."),
    ],
    "VIE": [
        ("Schonbrunn Palace", "landmark", 4.7, 2, "The Habsburgs' 1,441-room summer palace, with formal gardens and a zoo, Austria's most-visited sight."),
        ("St. Stephen's Cathedral", "religion", 4.7, 0, "A Gothic cathedral with a glazed tile roof, the spiritual heart of Vienna since the 12th century."),
        ("Belvedere Palace", "museum", 4.7, 1, "A Baroque palace complex holding Klimt's The Kiss and Austria's finest art collection."),
        ("Vienna State Opera", "culture", 4.7, 3, "One of the world's leading opera houses, offering standing-room tickets for a few euros."),
        ("Naschmarkt", "food", 4.4, 1, "Vienna's largest market, stalls of Middle Eastern, Balkan and Austrian food side by side."),
        ("Hofburg Palace", "landmark", 4.6, 1, "The former imperial palace, now housing the Spanish Riding School and Sisi Museum."),
        ("Prater Park", "nature", 4.3, 0, "A public park famed for its giant Ferris wheel and old-fashioned amusement rides."),
        ("MuseumsQuartier", "culture", 4.5, 1, "A cultural complex of modern art museums around courtyards full of design furniture."),
        ("Danube Canal Nightlife", "nightlife", 4.2, 0, "A stretch of bars, graffiti art and summer pop-up beaches along the canal."),
        ("Kunsthistorisches Museum", "museum", 4.7, 1, "An encyclopedic art museum with an unrivaled Bruegel collection, housed in a palatial building."),
    ],
    "LIS": [
        ("Belem Tower", "landmark", 4.5, 0, "A fortified 16th-century tower marking the departure point of Portugal's Age of Discoveries voyages."),
        ("Jeronimos Monastery", "landmark", 4.7, 1, "An ornate Manueline-style monastery built to commemorate Vasco da Gama's voyage to India."),
        ("Alfama District", "culture", 4.6, 0, "Lisbon's oldest quarter, a hillside tangle of alleys, fado bars and tiled facades."),
        ("Tram 28", "culture", 4.2, 0, "A vintage yellow tram that rattles through the steepest, most scenic parts of the old city."),
        ("LX Factory", "nightlife", 4.4, 1, "A converted industrial complex of bars, bookshops and street art under the 25 de Abril bridge."),
        ("Sao Jorge Castle", "landmark", 4.5, 0, "A Moorish hilltop castle with sweeping views over the terracotta rooftops of Lisbon."),
        ("Time Out Market", "food", 4.4, 1, "A curated food hall bringing together stalls from the city's best chefs and restaurants."),
        ("Praca do Comercio", "culture", 4.5, 0, "A grand riverside square that once served as the ceremonial entrance to the city."),
        ("Sintra Day Trip", "nature", 4.7, 1, "A fairy-tale hill town of pastel palaces and misty forests, a short train ride from Lisbon."),
        ("Cais do Sodre Nightlife", "nightlife", 4.3, 0, "The go-to district for late-night bars centered on the neon-lit Pink Street."),
    ],
}

# Opening hours by category (24h clock; close_hour < open_hour means the
# venue closes after midnight). Used by optimization/routing.py to make the
# router wait for opening or skip a stop that's closed for the rest of the
# day, instead of treating every POI as open around the clock.
OPENING_HOURS_BY_CATEGORY = {
    "museum": (9, 18), "landmark": (8, 20), "religion": (7, 19),
    "history": (8, 19), "culture": (9, 20), "shopping": (10, 20),
    "food": (10, 23), "nightlife": (18, 2), "nature": (0, 24), "beach": (0, 24),
}

poi_rows = []
poi_id = 1
for dest_id, pois in POI_TEMPLATES.items():
    dest_row = destinations_df[destinations_df.destination_id == dest_id].iloc[0]
    for name, category, rating, price_level, desc in pois:
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0.005, 0.05)  # ~0.5-5km scatter
        lat = dest_row.lat + radius * math.cos(angle)
        lon = dest_row.lon + radius * math.sin(angle) / math.cos(math.radians(dest_row.lat))
        open_hour, close_hour = OPENING_HOURS_BY_CATEGORY.get(category, (9, 18))
        poi_rows.append({
            "poi_id": f"POI{poi_id:04d}",
            "destination_id": dest_id,
            "name": name,
            "category": category,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "avg_visit_minutes": random.choice([45, 60, 90, 120, 150, 180]),
            "price_level": price_level,
            "popularity_score": rating,
            "description": desc,
            "open_hour": open_hour,
            "close_hour": close_hour,
        })
        poi_id += 1

poi_df = pd.DataFrame(poi_rows)
poi_df.to_csv(HERE / "poi.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Transport hubs
# ---------------------------------------------------------------------------
TRANSPORT = {
    "IST": [("Istanbul Airport", "airport", 41.2753, 28.7519), ("Sirkeci Train Station", "train_station", 41.0130, 28.9770)],
    "PAR": [("Charles de Gaulle Airport", "airport", 49.0097, 2.5479), ("Gare du Nord", "train_station", 48.8809, 2.3553)],
    "ROM": [("Fiumicino Airport", "airport", 41.8003, 12.2389), ("Roma Termini", "train_station", 41.9010, 12.5017)],
    "BCN": [("Barcelona El Prat Airport", "airport", 41.2974, 2.0833), ("Barcelona Sants", "train_station", 41.3792, 2.1400)],
    "AMS": [("Schiphol Airport", "airport", 52.3105, 4.7683), ("Amsterdam Centraal", "train_station", 52.3791, 4.9003)],
    "PRG": [("Vaclav Havel Airport", "airport", 50.1008, 14.2600), ("Praha hlavni nadrazi", "train_station", 50.0830, 14.4353)],
    "VIE": [("Vienna Intl Airport", "airport", 48.1103, 16.5697), ("Wien Hauptbahnhof", "train_station", 48.1853, 16.3775)],
    "LIS": [("Humberto Delgado Airport", "airport", 38.7813, -9.1359), ("Lisboa Oriente", "train_station", 38.7679, -9.0987)],
}
transport_rows = []
t_id = 1
for dest_id, hubs in TRANSPORT.items():
    for name, htype, lat, lon in hubs:
        transport_rows.append({
            "transport_id": f"TR{t_id:03d}",
            "destination_id": dest_id,
            "name": name,
            "type": htype,
            "lat": lat,
            "lon": lon,
        })
        t_id += 1
transport_df = pd.DataFrame(transport_rows)
transport_df.to_csv(HERE / "transport.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Monthly tourism demand time series (2019-01 .. 2026-06) per city
#    trend + annual seasonality + COVID-era shock + noise
# ---------------------------------------------------------------------------
dates = pd.date_range("2019-01-01", "2026-06-01", freq="MS")
base_levels = {  # rough relative popularity multiplier
    "IST": 1.3, "PAR": 1.6, "ROM": 1.4, "BCN": 1.2, "AMS": 1.0, "PRG": 0.8, "VIE": 0.9, "LIS": 0.9,
}
summer_peak_cities = {"BCN", "LIS", "AMS"}  # peak Jun-Aug harder
demand_rows = []
for dest_id in TRANSPORT.keys():
    base = 250_000 * base_levels[dest_id]
    trend_rate = random.uniform(0.0015, 0.004)  # monthly growth
    phase = 0 if dest_id in summer_peak_cities else random.choice([0, 1])
    for i, d in enumerate(dates):
        month = d.month
        seasonal_strength = 0.45 if dest_id in summer_peak_cities else 0.30
        seasonal = 1 + seasonal_strength * math.sin(2 * math.pi * (month - (6 if phase == 0 else 4)) / 12)
        trend = (1 + trend_rate) ** i
        covid_factor = 1.0
        if pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2021-05-01"):
            covid_factor = 0.08
        elif pd.Timestamp("2021-06-01") <= d <= pd.Timestamp("2022-06-01"):
            covid_factor = 0.5
        noise = np.random.normal(1.0, 0.05)
        visitors = max(0, base * trend * seasonal * covid_factor * noise)
        demand_rows.append({"destination_id": dest_id, "date": d.strftime("%Y-%m-%d"), "visitors": int(visitors)})
demand_df = pd.DataFrame(demand_rows)
demand_df.to_csv(HERE / "demand_timeseries.csv", index=False)

# ---------------------------------------------------------------------------
# 5. User preference archetypes (for segmentation training)
# ---------------------------------------------------------------------------
ARCHETYPES = {
    "Culture Enthusiast": dict(budget=0.6, culture=0.95, nature=0.3, nightlife=0.2, relax=0.3, adventure=0.2),
    "Beach & Relax": dict(budget=0.5, culture=0.2, nature=0.6, nightlife=0.3, relax=0.95, adventure=0.2),
    "Budget Backpacker": dict(budget=0.1, culture=0.5, nature=0.5, nightlife=0.6, relax=0.3, adventure=0.6),
    "Luxury Traveler": dict(budget=0.95, culture=0.6, nature=0.3, nightlife=0.4, relax=0.7, adventure=0.2),
    "Nightlife Seeker": dict(budget=0.5, culture=0.3, nature=0.2, nightlife=0.95, relax=0.2, adventure=0.5),
    "Nature & Adventure": dict(budget=0.5, culture=0.2, nature=0.9, nightlife=0.2, relax=0.3, adventure=0.9),
    "Family Traveler": dict(budget=0.6, culture=0.5, nature=0.6, nightlife=0.1, relax=0.6, adventure=0.3),
}
rows = []
for name, center in ARCHETYPES.items():
    for _ in range(60):
        rows.append({
            "archetype": name,
            **{k: float(np.clip(np.random.normal(v, 0.08), 0, 1)) for k, v in center.items()},
        })
survey_df = pd.DataFrame(rows)
survey_df.to_csv(HERE / "user_survey.csv", index=False)

# ---------------------------------------------------------------------------
# 6. City guide free-text corpus (for semantic + keyword retrieval)
# ---------------------------------------------------------------------------
CITY_GUIDES = {
    "IST": "Istanbul straddles Europe and Asia across the Bosphorus, layering Byzantine, Ottoman and modern eras in one skyline. "
           "The historic peninsula around Sultanahmet holds Hagia Sophia, the Blue Mosque, Topkapi Palace and the Basilica Cistern within "
           "walking distance of each other, making it the natural first-day zone for visitors arriving via Istanbul Airport. Cross the "
           "Golden Horn to Karakoy and Beyoglu for waterfront seafood, art galleries and the nightlife strip along Istiklal Street. Best "
           "visited April-May or September-October to avoid summer crowds and heat; Ramadan dates shift and affect restaurant hours. "
           "Budget travelers can rely on the extensive tram and ferry network instead of taxis. A half-day Bosphorus cruise pairs well "
           "with a Princes' Islands day trip for travelers with more time.",
    "PAR": "Paris rewards travelers who pace themselves by neighborhood rather than crisscrossing the city. The Eiffel Tower, Musee d'Orsay "
           "and Champs-Elysees cluster in the west; the Louvre, Notre-Dame and Le Marais sit closer together on the Right Bank and Ile de la "
           "Cite. Montmartre in the north offers cheaper eats and artist-quarter charm away from the main tourist flow. The city is compact "
           "enough to walk between arrondissements, and the metro covers any longer hops. Peak season is June-August with the longest queues "
           "at the Louvre and Eiffel Tower; a Versailles day trip requires a full day and advance timed tickets. Paris skews toward the luxury "
           "and mid-range budget tiers, especially around the Champs-Elysees.",
    "ROM": "Rome's ancient core -- the Colosseum, Roman Forum and Palatine Hill -- forms a single walkable archaeological zone best tackled "
           "in the cool morning hours. The Vatican, technically a separate city-state, requires a distinct half-day for the Museums, Sistine "
           "Chapel and St. Peter's Basilica, with lines that reward early or pre-booked entry. Trastevere across the Tiber is the place for "
           "evening trattorias and wine bars once the ruins close. Rome is busiest and hottest July-August; spring and early autumn give more "
           "comfortable walking weather. Termini station anchors the train and metro network and connects onward to Fiumicino Airport.",
    "BCN": "Barcelona splits neatly between Gaudi's Modernist landmarks -- Sagrada Familia, Park Guell and Casa Batllo -- and the medieval "
           "lanes of the Gothic Quarter down toward the harbor and Barceloneta Beach. The beach and the historic center are an easy walk or "
           "metro ride apart, letting a single day mix culture and relaxation. El Born, next to the Gothic Quarter, is the strongest nightlife "
           "and tapas zone. Sagrada Familia and Park Guell both require timed tickets booked well ahead in summer. Barcelona is a strong choice "
           "for travelers who want beach time without sacrificing museums and architecture, and it trends toward mid-range pricing.",
    "AMS": "Amsterdam's canal ring is the connective tissue of the city -- the Anne Frank House, Jordaan district and most museums sit within "
           "or just beside it, making the city one of the most walkable and bike-friendly in Europe. The Rijksmuseum and Van Gogh Museum "
           "anchor Museumplein, a short tram ride from the center, with Vondelpark next door for a break. The Red Light District and the "
           "Albert Cuyp Market give a grittier, more local flavor a few streets over. Rain is likely in any season, so indoor museum time "
           "should be spread across the itinerary rather than saved for one wet day. Amsterdam pairs well with a short list of days since "
           "most sights sit within a two-kilometer radius of Centraal Station.",
    "PRG": "Prague's Old Town, Charles Bridge and Prague Castle form a single scenic corridor across the Vltava, walkable in a day but best "
           "spread over two to linger. The castle complex is large enough to fill a half-day on its own, including St. Vitus Cathedral. "
           "Josefov, the historic Jewish Quarter, sits inside the Old Town loop. Prague remains one of the most budget-friendly major European "
           "capitals, with cheap beer and affordable museum tickets relative to Western Europe. Crowds peak in summer and around the Christmas "
           "markets; a dawn walk across Charles Bridge before the tour groups arrive is a favorite local tip.",
    "VIE": "Vienna's imperial core -- Hofburg Palace, St. Stephen's Cathedral and the MuseumsQuartier -- sits inside the Ringstrasse and is "
           "entirely walkable, with Schonbrunn Palace a short tram ride further out requiring its own half-day. The city has a strong "
           "classical music culture; standing-room tickets at the State Opera are inexpensive and available same-day. The Naschmarkt is the "
           "best food-focused stop, blending Austrian and Middle Eastern stalls. Vienna trends toward the luxury end of the budget spectrum "
           "but rewards it with exceptionally clean, efficient public transport and consistently high museum quality.",
    "LIS": "Lisbon is a city of hills, and the historic Alfama district is best explored on foot or via the iconic Tram 28 rather than a "
           "fixed walking route. Belem, with the Jeronimos Monastery and Belem Tower, sits west along the river and needs its own half-day "
           "trip by tram or train. Sintra, a fairy-tale hill town of palaces, is a popular full-day excursion by train from Rossio station. "
           "LX Factory and Cais do Sodre carry the nightlife scene in restored industrial and dockside spaces. Lisbon remains one of Western "
           "Europe's most budget-friendly capitals, with strong value in both food and accommodation.",
}
guides_dir = HERE / "city_guides"
guides_dir.mkdir(exist_ok=True)
for dest_id, text in CITY_GUIDES.items():
    (guides_dir / f"{dest_id}.txt").write_text(text)

print("Generated:")
print(f"  destinations.csv  ({len(destinations_df)} rows)")
print(f"  poi.csv           ({len(poi_df)} rows)")
print(f"  transport.csv     ({len(transport_df)} rows)")
print(f"  demand_timeseries.csv ({len(demand_df)} rows)")
print(f"  user_survey.csv   ({len(survey_df)} rows)")
print(f"  city_guides/*.txt ({len(CITY_GUIDES)} files)")
