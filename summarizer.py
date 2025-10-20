import argparse
from transformers import pipeline
import sys

def read_text_file(filepath):
    """
    Читает текст из файла.
    
    :param filepath: Путь к файлу
    :return: Текст из файла
    :raises IOError: При ошибках чтения
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # Проверка, что файл не пуст
        if not text.strip():
            raise ValueError("File is empty")
        return text
    except Exception as e:
        raise IOError(f"Error reading file '{filepath}': {e}")

def write_text_file(filepath, text):
    """
    Записывает текст в файл.
    
    :param filepath: Путь к файлу
    :param text: Текст для записи
    :raises IOError: При ошибках записи
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise IOError(f"Error writing to file '{filepath}': {e}")

def create_summarizer(model_name="facebook/bart-large-cnn"):
    """
    Создаёт pipeline для суммаризации, загружая модель.
    
    :param model_name: Имя модели для суммаризации
    :return: Объект pipeline для суммаризации
    """
    return pipeline('summarization', model=model_name)

def summarize_text(summarizer, text, max_length=50, min_length=25):
    """
    Формирует краткое резюме текста с использованием переданного summarizer.
    
    :param summarizer: pipeline для суммаризации
    :param text: Исходный текст
    :param max_length: Максимальная длина резюме
    :param min_length: Минимальная длина резюме
    :return: Резюме текста
    """
    # Вызов pipeline для создания резюме с заданными параметрами
    result = summarizer(text, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return result[0]['summary_text']

def main():
    parser = argparse.ArgumentParser(description="Text summarization with BART")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to output summary file")
    args = parser.parse_args()

    try:
        # Чтение входного текста
        text = read_text_file(args.input)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # Загрузка модели
    summarizer = create_summarizer()

    try:
        # Получение резюме текста
        summary = summarize_text(summarizer, text)
    except Exception as e:
        # Ошибка в процессе суммаризации
        print(f"Error during summarization: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        # Запись результата в выходной файл
        write_text_file(args.output, summary)
    except Exception as e:
        # Ошибка при записи файла
        print(e, file=sys.stderr)
        sys.exit(3)

    print(f"Summary successfully saved to '{args.output}'")

if __name__ == "__main__":
    main()
