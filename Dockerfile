FROM python:3.11-slim
ENV PORT 8000


COPY requirements.txt /
RUN pip install -r requirements.txt

RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_sm

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./Songs_Anlyzer /Songs_Anlyzer

CMD uvicorn Songs_Anlyzer.main:app --host 0.0.0.0 --port ${PORT}