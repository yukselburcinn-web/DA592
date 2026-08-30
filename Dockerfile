# RoamWise -- Streamlit interactive prototype
#
# Build:  docker build -t roamwise .
# Run:    docker run -p 8501:8501 roamwise
# Open:   http://localhost:8501
#
# The image ships the committed dataset (roamwise/data/*), so there is no
# data-generation step -- at build time or at runtime. The catalogue comes from
# Wikidata/OpenStreetMap/Wikipedia (#27), the demand series from Eurostat, and
# the knowledge graph is built in memory from those CSVs at startup.
#
# This comment used to call the dataset "synthetic" and tell you to run
# `data/generate_data.py` for fresh data (#145). Both were wrong, and the
# second is destructive: that script predates the two-city migration and
# rewrites five shipped files with the old eight-city set, which does not
# start -- two destinations against eight city guides raises KeyError in
# retrieval/corpus.py. To rebuild data, see README "Rebuilding the datasets",
# which uses roamwise/pipeline/ and not that script.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY roamwise/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY roamwise/ ./roamwise/
WORKDIR /app/roamwise

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
