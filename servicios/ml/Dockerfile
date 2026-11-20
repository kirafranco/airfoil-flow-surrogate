# GPU job runner for every stage of the pipeline: ingestion, ETL, training,
# evaluation and the test suite. It is run, not started:
#
#   docker compose --profile ml run --rm ml                      # test suite
#   docker compose --profile ml run --rm ml python -m <module>   # a pipeline stage
#
# Built on the official python image rather than nvidia/cuda or pytorch/pytorch:
# the PyPI torch wheel bundles its own CUDA runtime (cuda-toolkit 13.2 at these
# pins), so the host only has to provide the NVIDIA driver, which Docker
# Desktop's WSL2 backend passes through.
#
# The instructions up to the marker below are shared, byte for byte, with
# servicios/jupyter/Dockerfile, so BuildKit stores the ~3 GB dependency layer
# once for both images. tests/test_stack_compose.py fails if they drift apart.
FROM python:3.12.16-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/project \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies before code, so editing a module never reinstalls torch.
# `pip check` fails the build on a pin conflict instead of at import time.
COPY requirements.txt requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-dev.txt \
    && pip check

# ---- end of the prefix shared with servicios/jupyter/Dockerfile ----

# A named account for uid 1000, not just the number, with a real HOME: a bare
# `USER 1000:1000` has no passwd entry, and tooling that resolves the current
# user fails on it. uid 1000 keeps bind-mounted files from being root-owned.
RUN useradd --uid 1000 --create-home --home-dir /home/app --shell /usr/sbin/nologin app
ENV HOME=/home/app

# The image carries the code so it runs standalone. Compose mounts
# the live project over it read-only, which also brings tests/ and pytest.ini.
COPY src/ /opt/project/src/
WORKDIR /opt/project

USER 1000:1000

CMD ["python", "-m", "pytest"]
