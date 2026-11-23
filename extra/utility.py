import re
import numpy as np


def create_unique_word_dict(text: list) -> dict:
    """
    Eine Methode, die ein Dictionary erstellt, in dem die Schlüssel die verschiedenen Wörter
    und die Werte Indizes sind
    """
    # Alle verschiedenen Wörter aus unserem Text holen und alphabetisch sortieren
    words = list(set(text))
    words.sort()

    # Das Dictionary für die verschiedenen Wörter erstellen
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({word: i})

    return unique_word_dict


def text_preprocessing(
    text: list,
    punctuations=r"""!()-[]{};:'"\,<>./?@#$%^&*_“~""",
    stop_words=["and", "a", "is", "the", "in", "be", "will"],
) -> list:
    """
    Eine Methode, um Text vorzuverarbeiten
    """
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, "")

    # Wörter entfernen, die Zahlen enthalten
    text = re.sub(r"\w*\d\w*", "", text)

    # Ziffern entfernen
    text = re.sub(r"[0-9]+", "", text)

    # Leerzeichen bereinigen
    text = re.sub(r"\s+", " ", text).strip()

    # Jedes Wort in Kleinbuchstaben umwandeln
    text = text.lower()

    # Unseren ganzen Text in eine Liste umwandeln
    text = text.split(" ")

    # Leere Strings entfernen
    text = [x for x in text if x != ""]

    # Stoppwörter entfernen
    text = [x for x in text if x not in stop_words]

    return text


# Funktionen, um das ähnlichste Wort zu finden
def euclidean(vec1: np.array, vec2: np.array) -> float:
    """
    Eine Funktion, die den euklidischen Abstand zwischen zwei Vektoren berechnet
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def find_similar(word: str, embedding_dict: dict, top_n=10) -> list:
    """
    Eine Methode, die anhand der gelernten Embeddings das ähnlichste Wort findet
    """
    dist_dict = {}
    word_vector = embedding_dict.get(word, [])
    if len(word_vector) > 0:
        for key, value in embedding_dict.items():
            if key != word:
                dist = euclidean(word_vector, value)
                dist_dict.update({key: dist})

        return sorted(dist_dict.items(), key=lambda x: x[1])[0:top_n]
