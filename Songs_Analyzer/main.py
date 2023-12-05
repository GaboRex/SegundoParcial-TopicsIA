from fastapi import FastAPI, HTTPException
from typing import List
from fastapi.responses import FileResponse
from pydantic import BaseModel
from Songs_Analyzer.report_generator import SentimentReportGenerator
from Songs_Analyzer.status import get_status
from Songs_Analyzer.sentiment import analyze_sentiment
from Songs_Analyzer.sentiment_analysis import extract_title_and_artist, perform_spacy_analysis
from newspaper import Article


app = FastAPI(title="Song Sentiment Analyzer")

class SongAnalysis(BaseModel):
    urls: List[str]

predictions = []
report_generator = SentimentReportGenerator()


@app.get("/status")
def get_app_status():
    return get_status()

@app.post("/sentiment")
def analyze_sentiment_endpoint(song_data: SongAnalysis, language: str = 'es'):
    urls = song_data.urls

    if not urls:
        raise HTTPException(status_code=400, detail="La lista de URLs no puede estar vacía.")

    results = []

    for url in urls:
        metadata = extract_title_and_artist(url, language=language)

        if not metadata["info"]:
            raise HTTPException(status_code=400, detail=f"No se pudo extraer información de la URL: {url}")

        article = Article(url)
        article.download()
        article.parse()
        lyrics = article.text

        if not lyrics.strip():
            raise HTTPException(status_code=400, detail=f"No se pudo extraer texto de la URL: {url}")

        sentiment_info = analyze_sentiment(lyrics, language=language)
        results.append({"url": url, "info": metadata["info"], "sentiment": sentiment_info})
        report_generator.add_prediction({"url": url, "info": metadata["info"], "sentiment": sentiment_info})

    return results

@app.post("/analysis")
def detailed_analysis_endpoint(song_data: SongAnalysis, language: str = 'es'):
    urls = song_data.urls

    if not urls:
        raise HTTPException(status_code=400, detail="La lista de URLs no puede estar vacía.")

    results = []

    for url in urls:
        metadata = extract_title_and_artist(url, language=language)

        if not metadata["info"]:
            raise HTTPException(status_code=400, detail=f"No se pudo extraer información de la URL: {url}")

        article = Article(url)
        article.download()
        article.parse()
        lyrics = article.text

        if not lyrics.strip():
            raise HTTPException(status_code=400, detail=f"No se pudo extraer texto de la URL: {url}")

        sentiment_info = analyze_sentiment(lyrics, language=language)
        spacy_analysis = perform_spacy_analysis(lyrics, language=language)

        results.append({"url": url, "info": metadata["info"], "sentiment": sentiment_info, "spacy_analysis": spacy_analysis})

    return results
    
@app.get("/reports")
def generate_reports():
    report_file = report_generator.generate_csv_report()

    if report_file is None:
        raise HTTPException(status_code=404, detail="No hay predicciones disponibles para generar un informe.")

    return FileResponse(report_file, media_type="text/csv", filename=report_file)
