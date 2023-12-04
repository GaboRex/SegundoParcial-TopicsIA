from fastapi import FastAPI, HTTPException
from analysis_sentiment_spacy import (
    SongAnalysis,
    analyze_sentiment_endpoint,
    detailed_analysis_endpoint,
)
from status import get_status  

app = FastAPI(title="Song Sentiment Analyzer")


@app.get("/status")
def get_app_status():
    return get_status() 


@app.post("/Sentiment")
def analyze_sentiment(song_data: SongAnalysis, language: str = 'es'):
    return analyze_sentiment_endpoint(song_data, language)


@app.post("/analysis")
def detailed_analysis(song_data: SongAnalysis, language: str = 'es'):
    return detailed_analysis_endpoint(song_data, language)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
