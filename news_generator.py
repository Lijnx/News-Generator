"""
news_generator.py
================

Система генерации новостей по факту.
Алгоритм:
1. Чтение исходного текста (описания события).
2. Извлечение ключевых слов (YAKE).
3. Генерация новостной статьи (llama3.1).
4. Создание заголовка с включением ключевых слов (mT5).
5. Сохранение результата в JSON.

Запуск:
    python news_generator.py --input event.txt --output news.json --lang ru
"""

import argparse
import json
import sys
import requests
from text_summarizer.summarizer import read_text_file
from title_with_keywords.task2.genai_title_with_keywords import keywords, build_prompt, summarize_title

# === Стандартная астройка модели генерации ===
DEFAULT_GEN_MODEL = "llama3.1"
DEFAULT_MAX_LEN = 1000

# === Стандартная настройка модели суммаризации ===
DEFAULT_SUMM_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
DEFAULT_MAX_TITLE_LEN = 32
DEFAULT_MIN_TITLE_LEN = 6

# === Интерфейс для работы с Ollama ===
def ask_ollama(prompt, model=DEFAULT_GEN_MODEL, temperature=0.7, num_predict=DEFAULT_MAX_LEN):
    """Запрос к локальной модели Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "options": {
                    "num_predict": num_predict
                }
            },
            timeout=120
        )

        return response.json()["response"]

    except Exception as e:
        return f"Ошибка: {e}"
    

# === Генерация статьи ===
def generate_article(event_text, keywords, model = DEFAULT_GEN_MODEL, lang: str = "ru", temperature = 0.7, num_predict=DEFAULT_MAX_LEN):
    """Создаёт новостную статью с встраиванием ключевых слов."""
    prompt = ""
    if lang.startswith("ru"):
        prompt = (
            f"Напиши новостную статью на русском языке на основе следующего события: {event_text}. "
            f"Обязательно включи в текст ключевые слова: {', '.join(keywords)}. "
            f"Стиль — новостной, нейтральный, связный, без комментариев и мнений."
        )
    else:
        prompt = (
            f"Write a news article based on the following event: {event_text}."
            f"Be sure to include keywords in the text: {', '.join(keywords)}."
            f"The style is news-like, neutral, coherent, without commentary or opinion."
        )

    return ask_ollama(prompt, model, temperature, num_predict)


# === Генерация заголовка ===
def generate_title(article, keywords_list, lang:str = "ru"):
    """Создаёт заголовок статьи."""
    prompt = build_prompt(keywords_list, lang)
    return summarize_title(article, prompt, DEFAULT_SUMM_MODEL, DEFAULT_MIN_TITLE_LEN, DEFAULT_MAX_TITLE_LEN)


# === Основной workflow ===
def main():
    parser = argparse.ArgumentParser(description="News generator")
    parser.add_argument("--input", type=str, required=True, help="Путь к файлу с описанием события")
    parser.add_argument("--output", type=str, required=True, help="Путь к JSON файлу для сохранения результата")
    parser.add_argument("--lang", type=str, default="ru", help="Язык анализа и генерации")
    parser.add_argument("--num_keywords", type=int, default=6, help="Количество ключевых слов")
    parser.add_argument("--gen_model", type=str, default=DEFAULT_GEN_MODEL, help="Модель Ollama для генерации статьи")
    parser.add_argument("--temperature", type=float, default=0.7, help="Диапазон: 0.0 - 1.0. Контролирует случайность/креативность статьи")
    parser.add_argument("--num_predict", type=int, default=DEFAULT_MAX_LEN, help="Макс длина статьи (токены)")
    args = parser.parse_args()

    try:
        event_text = read_text_file(args.input)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    keywords_list = keywords(event_text, lang=args.lang, keywords_num=args.num_keywords)
    if not keywords_list:
        print("Не удалось извлечь ключевые слова", file=sys.stderr)
        sys.exit(2)

    # Генерация новостной статьи
    article = generate_article(event_text, keywords_list, args.gen_model, args.lang, args.temperature, args.num_predict)

    # Генерация заголовка
    title = generate_title(article, keywords_list, args.lang)

    # Сохранение в JSON
    result = {
        "event": event_text,
        "title": title,
        "article": article,
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Новость успешно сохранена в {args.output}")
    except Exception as e:
        print(f"Ошибка записи JSON: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
