FROM python:3.10-slim

WORKDIR /app

# Install uv and Rust (required for uv)
RUN apt-get update && apt-get install -y curl build-essential \
    && curl -sSf https://install.astral.sh/uv/install.sh | bash

ENV PATH="/root/.cargo/bin:${PATH}"

COPY . .

RUN uv pip install

EXPOSE 8501

CMD [ "python", "02_app.py" ]