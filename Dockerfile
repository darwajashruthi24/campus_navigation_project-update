FROM python:3.11-slim
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# copy project
COPY . /app

# install python deps
RUN python -m pip install --upgrade pip
RUN if [ -f requirements-gradio.txt ]; then pip install -r requirements-gradio.txt; fi

EXPOSE 7860

CMD ["python", "gradio_app.py"]
