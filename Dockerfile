FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY robotprobe ./robotprobe
COPY robotsec_range ./robotsec_range
COPY interventions ./interventions
COPY examples ./examples
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir '.[ros2-decoders]'

RUN useradd --create-home --uid 1000 probe
USER probe

CMD ["robotprobe", "--help"]
