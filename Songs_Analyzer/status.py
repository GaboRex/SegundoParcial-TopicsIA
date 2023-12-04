
def get_status():
    return {
        "status": "Servicio en funcionamiento",
        "models": {
            "espanol": {
                "model_name": "karina-aquino/spanish-sentiment-model",
                "status": "Cargado y listo para su uso",
            },
            "english": {
                "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
                "status": "Cargado y listo para su uso",
            },
            "spacy_es": {"model_name": "es_core_news_md", "status": "Cargado y listo para su uso"},
            "spacy_en": {"model_name": "en_core_web_sm", "status": "Cargado y listo para su uso"},
        },
    }
