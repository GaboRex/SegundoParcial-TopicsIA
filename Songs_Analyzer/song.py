from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from newspaper import Article
from bs4 import BeautifulSoup
import spacy
import requests

app = FastAPI(title="Song Sentiment Analyzer")

class SongAnalysis(BaseModel):
    url: str

# Cargar el modelo en español una vez al inicio de la aplicación
model_es = AutoModelForSequenceClassification.from_pretrained('karina-aquino/spanish-sentiment-model')
tokenizer_es = AutoTokenizer.from_pretrained('karina-aquino/spanish-sentiment-model')
sentiment_analyzer_es = pipeline('sentiment-analysis', model=model_es, tokenizer=tokenizer_es)

# Cargar el modelo en inglés una vez al inicio de la aplicación
model_en = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer_en = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_analyzer_en = pipeline('sentiment-analysis', model=model_en, tokenizer=tokenizer_en)

# Cargar el modelo de spaCy para el procesamiento de texto en inglés y español
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
    # Seleccionar el modelo y tokenizer según el idioma
    model, tokenizer, analyzer = (model_es, tokenizer_es, sentiment_analyzer_es) if language == 'es' else (model_en, tokenizer_en, sentiment_analyzer_en)
    
    # Realizar el análisis de sentimientos
    result = analyzer(text)

    # Obtener la puntuación de estrellas de la salida del modelo
    star_score = result[0]['score']  # Asumiendo que 'score' es una puntuación de 1 a 5

    # Definir umbrales para categorías de sentimientos
    threshold_negative = 0.2
    threshold_positive = 0.8

    # Mapear la puntuación de estrellas a una etiqueta de sentimiento
    if star_score < threshold_negative:
        sentiment_label = 'muy negativo'
    elif threshold_negative <= star_score < 0.5:
        sentiment_label = 'negativo'
    elif 0.5 <= star_score < threshold_positive:
        sentiment_label = 'neutral'
    elif threshold_positive <= star_score < 1:
        sentiment_label = 'positivo'
    else:
        sentiment_label = 'muy positivo'

    return {"score": star_score, "sentiment": sentiment_label}



def extract_title_and_artist(url, language='es'):
    article = Article(url)
    article.download()
    article.parse()

    # Obtener el texto del artículo
    text = article.text

    # Seleccionar el modelo de spaCy según el idioma
    nlp = nlp_es if language == 'es' else nlp_en

    # Procesar el texto con spaCy
    doc = nlp(text)

    # Buscar entidades de tipo 'TITLE'
    title_entities = [ent.text for ent in doc.ents if ent.label_ == 'TITLE']

    # Tomar el primer título encontrado
    title = title_entities[0] if title_entities else None

    # Si no se pudo obtener el título con spaCy, intentar con BeautifulSoup
    if not title:
        title = extract_title_with_beautifulsoup(url)

    return {"info": title}

def extract_title_with_beautifulsoup(url):
    # Realizar una solicitud HTTP para obtener el contenido HTML de la página
    response = requests.get(url)
    html_content = response.text

    # Utilizar BeautifulSoup para analizar el HTML y encontrar el título
    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')

    # Si se encuentra la etiqueta de título, extraer el texto
    if title_tag:
        return title_tag.text

    return None

@app.post("/analyze_sentiment")
def analyze_sentiment_endpoint(song_data: SongAnalysis, language: str = 'es'):
    url = song_data.url

    if not url:
        raise HTTPException(status_code=400, detail="La URL no puede estar vacía.")

    # Extraer el título de la URL
    metadata = extract_title_and_artist(url, language=language)

    if not metadata["info"]:
        raise HTTPException(status_code=400, detail="No se pudo extraer información de la URL proporcionada.")

    # Extraer el texto de la URL
    article = Article(url)
    article.download()
    article.parse()
    lyrics = article.text

    if not lyrics.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto de la URL proporcionada.")

    # Procesar la letra antes de enviarla al modelo
    sentiment = analyze_sentiment(lyrics, language=language)

    return {"info": metadata["info"], "sentiment": sentiment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("song:app", reload=True)
