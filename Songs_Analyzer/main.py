from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from status import get_status
from sentiment import analyze_sentiment
from sentiment_analysis import extract_title_and_artist, perform_spacy_analysis
from newspaper import Article


app = FastAPI(title="Song Sentiment Analyzer")

class SongAnalysis(BaseModel):
    urls: List[str]

predictions = []


@app.get("/status")
def get_app_status():
    return get_status()

@app.post("/Sentiment")
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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)