# clashBot demo image (MAC-AV): the live stack in dry run on a bundled clip, plus the decision dashboard.
#
# Build context = the root of the Docker submission zip (scripts/make_docker_submission.sh): clashbot/, cr_perception/,
# configs/, knowledge_base/, tools/, tests/, docker/, artifacts/ (KataCR weights, databases, the clip) and prior_runs/ (the
# prior games the dashboard can browse). Multi-arch: the same
# file builds linux/amd64 and linux/arm64 natively on the grader's machine. The d1 ranker (1.17 GB) is downloaded from
# Hugging Face during the build (needs internet); everything else is in the zip.
#
# Layer order, slowest-changing first: system packages -> torch (CPU) -> the other Python packages -> d1 -> artefacts -> code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV (libgl1, libglib2.0-0) and OpenMP (libgomp1: onnxruntime, LightGBM)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# torch 2.15.0 as pinned in requirements-mac.txt; torchvision 0.29.1 (the CPU index has no 0.29.2). CPU wheels exist for amd64 and arm64.
RUN pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
        torch==2.15.0+cpu torchvision==0.29.1+cpu

# The other packages. build-essential only for hnswlib (sdist) and is removed again in the same layer.
# ultralytics and rapidocr-onnxruntime go in without their dependencies: they declare the GUI build of OpenCV.
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install -r /tmp/requirements-docker.txt \
 && pip install --no-deps ultralytics==8.1.24 rapidocr-onnxruntime==1.4.4 \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && python -c "import torch, torchvision, cv2, onnxruntime, rapidocr_onnxruntime, transformers, soundfile, ultralytics.nn.tasks, ultralytics.utils.loss, ultralytics.utils.tal; \
from torchvision.ops import nms; print('python packages ok: torch', torch.__version__, 'torchvision', torchvision.__version__, 'cv2', cv2.__version__)"

WORKDIR /app

# d1: the fine-tuned ranker (fp16 safetensors, 1.17 GB; code files, tokenizer and the LFM Open License come with it) from the
# public Hugging Face repo Jyler/clashbot-d1-ft-joint-tac, pinned to one revision. No token. Its own layer, after the packages and
# before the artefacts and the code, so editing code never downloads it again. HF_HUB_OFFLINE is set only after this layer.
# A download that fails three times does not fail the build: the folder is removed and the container falls back to the tactics
# engine (with a banner). Retry just this layer with: D1_RETRY=1 docker compose up --build
ARG D1_RETRY=0
RUN echo "d1 download (retry token $D1_RETRY)"; \
    for attempt in 1 2 3; do \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('Jyler/clashbot-d1-ft-joint-tac', revision='5b984f0cce7c6db6e8a768cc0df2d6836022a3e2', local_dir='/app/artifacts/d1-ft-joint-tac')" && break; \
      echo "d1 download attempt $attempt failed"; sleep 10; \
    done; \
    if [ -f /app/artifacts/d1-ft-joint-tac/model.safetensors ]; then echo "d1 weights downloaded"; \
    else echo "WARNING: d1 weights could not be downloaded; the image will use the tactics engine (retry: D1_RETRY=1 docker compose up --build)"; \
      rm -rf /app/artifacts/d1-ft-joint-tac; fi

ENV HF_HUB_OFFLINE=1 \
    CLASHBOT_TORCH_DEVICE=cpu \
    CLASHBOT_KATACR_ROOT=/app/artifacts/katacr \
    CLASHBOT_KATACR_WEIGHTS=/app/artifacts/katacr_d1.pt:/app/artifacts/katacr_d2.pt \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Artefacts: KataCR weights and source, the databases, the act model, the clip. clashbot.config looks for
# db/ and dataset/ next to the code, hence the two links.
COPY artifacts/ /app/artifacts/
RUN ln -s /app/artifacts/db /app/db && ln -s /app/artifacts/dataset /app/dataset

# Prior games (MAC-AW): every live game, the bot-vs-pro runs and two dry runs on the test clip, screenshots re-encoded smaller
# (scripts/pack_prior_runs.py), at prior_runs/<group>/<run>/ in the zip. The files are read-only (0444) and nothing writes here;
# the dashboard lists them in its run picker beside the demo run (env CLASHBOT_VIZ_PARENTS is read by `python -m clashbot.viz`).
COPY prior_runs/ /app/prior_runs/
ENV CLASHBOT_VIZ_PARENTS=/app/prior_runs/poc,/app/prior_runs/demo,/app/prior_runs/vs_pro,/app/prior_runs/smoke

# Code and the reference data the runtime reads (card tables and art, hero index)
COPY configs/ /app/configs/
COPY knowledge_base/ /app/knowledge_base/
COPY tools/ /app/tools/
COPY cr_perception/ /app/cr_perception/
COPY clashbot/ /app/clashbot/
COPY tests/ /app/tests/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && mkdir -p /data/runs \
 && python -c "import clashbot.live, clashbot.viz.__main__; print('clashbot imports ok')"

VOLUME /data/runs
EXPOSE 8765
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["demo"]
