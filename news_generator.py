# good luck
import json
from transformers import pipeline
from summarizer import read_text_file, create_summarizer, summarize_text
from genai_title_with_keywords import keywords, coverage_score, build_prompt, summarize_title, coverage_score, integrate_missing_inline

event = read_text_file("text.txt")
kw = keywords(event, lang="ru", keywords_num=6)

# article creation

summarizer = create_summarizer(model_name="google/flan-t5-base")
article = summarize_text(
    summarizer,
    f"Сгенерируй новостную статью на основе события: {event}. "
    f"Включи ключевые слова: {', '.join(kw)}",
    max_length=400, min_length=150
)

#title creation

title_prompt = build_prompt(kw, lang="ru")
title = summarize_title(article, title_prompt)

cov, missed = coverage_score(article, kw)
if cov < 0.8:
    title = integrate_missing_inline(article, missed, kw)

news_item = {
    "title": title,
    "event": event,
    "keywords": kw,
    "article": article
}

with open("news.json", "a", encoding="utf-8") as f:
    json.dump(news_item, f, ensure_ascii=False, indent=2)
