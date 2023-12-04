from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from newspaper import Article
from bs4 import BeautifulSoup
import spacy
import time
import requests

app = FastAPI(title="Song Sentiment Analyzer")

class SongAnalysis(BaseModel):
    url: str

model_es = AutoModelForSequenceClassification.from_pretrained('karina-aquino/spanish-sentiment-model')
tokenizer_es = AutoTokenizer.from_pretrained('karina-aquino/spanish-sentiment-model')
sentiment_analyzer_es = pipeline('sentiment-analysis', model=model_es, tokenizer=tokenizer_es)

model_en = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer_en = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_analyzer_en = pipeline('sentiment-analysis', model=model_en, tokenizer=tokenizer_en)

nlp_es = spacy.load("es_core_news_md")
nlp_en = spacy.load("en_core_web_sm")

@app.get("/status")
def get_status():
    return {
        "status": "Servicio en funcionamiento",
        "models": {
            "espanol": {
                "model_name": "karina-aquino/spanish-sentiment-model",
                "status": "Cargado y listo para su uso"
            },
            "english": {
                "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
                "status": "Cargado y listo para su uso"
            },
            "spacy_es": {
                "model_name": "es_core_news_md",
                "status": "Cargado y listo para su uso"
            },
            "spacy_en": {
                "model_name": "en_core_web_sm",
                "status": "Cargado y listo para su uso"
            }
        }
    }

def analyze_sentiment(text, language='es'):
    start_time = time.time()

    model, tokenizer, analyzer = (model_es, tokenizer_es, sentiment_analyzer_es) if language == 'es' else (model_en, tokenizer_en, sentiment_analyzer_en)

    result = analyzer(text)
    score_normalized = (result[0]['score'] - 0.5) * 2

    threshold_negative = -0.5
    threshold_positive = 0.5

    if score_normalized < threshold_negative:
        sentiment_label = 'muy negativo'
    elif threshold_negative <= score_normalized < 0:
        sentiment_label = 'negativo'
    elif 0 <= score_normalized < threshold_positive:
        sentiment_label = 'neutral'
    elif threshold_positive <= score_normalized <= 1:
        sentiment_label = 'positivo'
    else:
        sentiment_label = 'muy positivo'

    execution_time = time.time() - start_time

    return {"score": score_normalized, "sentiment": sentiment_label, "execution_time": execution_time}

def extract_title_and_artist(url, language='es'):
    article = Article(url)
    article.download()
    article.parse()

    text = article.text
    nlp = nlp_es if language == 'es' else nlp_en

    doc = nlp(text)
    title_entities = [ent.text for ent in doc.ents if ent.label_ == 'TITLE']
    title = title_entities[0] if title_entities else None

    if not title:
        title = extract_title_with_beautifulsoup(url)

    return {"info": title}

def extract_title_with_beautifulsoup(url):
    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')

    if title_tag:
        return title_tag.text

    return None

def perform_spacy_analysis(text, language='es'):
    nlp = nlp_es if language == 'es' else nlp_en
    doc = nlp(text)

    pos_tags = [{'text': token.text, 'pos': token.pos_} for token in doc]
    ner_tags = [{'text': ent.text, 'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_} for ent in doc.ents]

    # Embeddings: promedio de vectores de palabras en el documento
    embedding = doc.vector.tolist()

    return {"pos_tags": pos_tags, "ner_tags": ner_tags, "embedding": embedding}

@app.post("/Sentiment")
def analyze_sentiment_endpoint(song_data: SongAnalysis, language: str = 'es'):
    url = song_data.url

    if not url:
        raise HTTPException(status_code=400, detail="La URL no puede estar vacía.")

    metadata = extract_title_and_artist(url, language=language)

    if not metadata["info"]:
        raise HTTPException(status_code=400, detail="No se pudo extraer información de la URL proporcionada.")

    article = Article(url)
    article.download()
    article.parse()
    lyrics = article.text

    if not lyrics.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto de la URL proporcionada.")

    sentiment_info = analyze_sentiment(lyrics, language=language)

    return {"info": metadata["info"], "sentiment": sentiment_info}

@app.post("/analysis")
def detailed_analysis_endpoint(song_data: SongAnalysis, language: str = 'es'):
    url = song_data.url

    if not url:
        raise HTTPException(status_code=400, detail="La URL no puede estar vacía.")

    metadata = extract_title_and_artist(url, language=language)

    if not metadata["info"]:
        raise HTTPException(status_code=400, detail="No se pudo extraer información de la URL proporcionada.")

    article = Article(url)
    article.download()
    article.parse()
    lyrics = article.text

    if not lyrics.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto de la URL proporcionada.")

    sentiment_info = analyze_sentiment(lyrics, language=language)
    spacy_analysis = perform_spacy_analysis(lyrics, language=language)

    return {"info": metadata["info"], "sentiment": sentiment_info, "spacy_analysis": spacy_analysis}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("song:app", reload=True)