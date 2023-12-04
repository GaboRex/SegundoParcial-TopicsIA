from newspaper import Article
from bs4 import BeautifulSoup
import spacy
import requests

nlp_es = spacy.load("es_core_news_md")
nlp_en = spacy.load("en_core_web_sm")

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

    embedding = doc.vector.tolist()

    return {"pos_tags": pos_tags, "ner_tags": ner_tags, "embedding": embedding}
