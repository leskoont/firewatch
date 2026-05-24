FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# the PyPI opencv wheels for Linux can't encode H.264, so clips go straight to mp4v
ENV FIREWATCH_CLIP_CODECS=mp4v

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY pyproject.toml README.md ./
RUN mkdir firewatch && touch firewatch/__init__.py \
    && pip install ".[cpu]" \
    && rm -rf firewatch build

COPY firewatch ./firewatch
RUN pip install --no-deps . && rm -rf build

COPY configs ./configs

RUN useradd --create-home --uid 1000 app \
    && mkdir -p events models \
    && chown app:app events
USER app

EXPOSE 8000
ENTRYPOINT ["firewatch"]
CMD ["run", "-c", "configs/cameras.yaml"]
