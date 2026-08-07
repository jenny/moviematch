FROM python:3.11-slim

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-bake the SentenceTransformers encoder (~418MB) into the image.
#
# Without this the weights download at *runtime*, on the first get_model() call — which
# the api/app.py lifespan triggers at startup to keep the first user query off the cold
# path. That put a 418MB anonymous HuggingFace Hub download on every cold start, subject
# to the unauthenticated rate limit, and made startup depend on huggingface.co being
# reachable for something we don't otherwise need at runtime.
#
# Deliberately kept ABOVE `COPY . .` so editing app code does not invalidate this layer.
# That means the model name can't be read from config.py (not copied yet), so it is
# pinned here and tests/test_dockerfile.py asserts it stays equal to config.MODEL_NAME.
ARG EMBEDDING_MODEL=all-mpnet-base-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# The weights are in the image cache from here on, so forbid runtime Hub calls entirely:
# a cold start can never stall on huggingface.co, and a model-name drift fails loudly at
# startup instead of silently re-downloading. Also silences the "unauthenticated requests
# to the HF Hub / set a HF_TOKEN" warning, since no such request is made.
# Must come AFTER the prefetch above — set any earlier and it would block that download.
ENV HF_HUB_OFFLINE=1

COPY . .

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
