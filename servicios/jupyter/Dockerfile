# JupyterLab for exploration only. Pipelines never
# import from notebooks; anything used twice moves into src/.
#
# Built on the project's own pins rather than a jupyter/* or quay.io/jupyter
# image, so that a kernel runs exactly the versions src/ and the tests do.
#
# The instructions up to the marker below are shared, byte for byte, with
# servicios/ml/Dockerfile, so BuildKit reuses the ml image's ~3 GB dependency
# layer instead of storing it twice. tests/test_stack_compose.py fails if they
# drift apart. The test tools come along with that layer, which is harmless.
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

# ---- end of the prefix shared with servicios/ml/Dockerfile ----

RUN pip install --no-cache-dir jupyterlab==4.6.6 \
    && pip check

# Named uid 1000 account with a real HOME (see servicios/ml/Dockerfile). The
# kernel runtime directory stays inside the container: connection files and
# ZeroMQ sockets are small-file churn that crawls on a Windows bind mount.
RUN useradd --uid 1000 --create-home --home-dir /home/jupyter --shell /usr/sbin/nologin jupyter \
    && mkdir -p /tmp/jupyter-runtime /opt/project/notebooks \
    && chown -R 1000:1000 /tmp/jupyter-runtime /home/jupyter /opt/project/notebooks
ENV HOME=/home/jupyter \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime

# The image carries the code so it runs standalone; compose mounts
# the live project over it.
COPY src/ /opt/project/src/

USER 1000:1000

EXPOSE 8888

# The file tree in the browser is the bind-mounted, versioned notebooks/ folder.
# The token comes from JUPYTER_TOKEN in the environment, set from .env.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--notebook-dir=/opt/project/notebooks"]
