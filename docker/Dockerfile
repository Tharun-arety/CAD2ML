# CAD2ML runtime image (api + worker). CPU only, non-root.
FROM mambaorg/micromamba:2.9.0-ubuntu24.04

USER root
# Runtime shared libraries needed by OCCT/VTK wheels on a headless host (no X server is started).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6 \
    libsm6 curl && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
    CAD2ML_DATA_DIR=/data CAD2ML_CORPUS_DIR=/data/corpus
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md alembic.ini ./
COPY --chown=$MAMBA_USER:$MAMBA_USER cad2ml ./cad2ml
COPY --chown=$MAMBA_USER:$MAMBA_USER apps ./apps
COPY --chown=$MAMBA_USER:$MAMBA_USER db ./db
COPY --chown=$MAMBA_USER:$MAMBA_USER schemas ./schemas
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts ./scripts
COPY --chown=$MAMBA_USER:$MAMBA_USER tests ./tests
RUN pip install --no-deps --no-build-isolation -e . && python scripts/spike_occt.py

USER root
RUN mkdir -p /data && chown $MAMBA_USER:$MAMBA_USER /data
USER $MAMBA_USER
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
