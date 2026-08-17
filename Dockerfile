# RoamWise -- Streamlit interactive prototype
#
# Build:  docker build -t roamwise .
# Run:    docker run -p 8501:8501 roamwise
# Open:   http://localhost:8501
#
# The image ships the committed synthetic dataset and pre-built knowledge
# graph (roamwise/data/*), so no data-generation step is needed at runtime.
# To use fresh/real data instead, run `python data/generate_data.py &&
# python knowledge_graph/build_graph.py` before building, or as a container
# entrypoint override.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY roamwise/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY roamwise/ ./roamwise/
WORKDIR /app/roamwise

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
