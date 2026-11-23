# Einführung in NLP

Eine praxisnahe Einführung in Natural Language Processing. Du beginnst mit einem Spamklassifikator aus dem klassischen Machine Learning, gehst dann zu modernen Transformer-Modellen für die Zero-Shot-Klassifikation über und erstellst zum Schluss Word Embeddings mit einem neuronalen Netz. Die Aufgaben verwenden den Datensatz SMS Spam Collection und vortrainierte Modelle aus der Transformers-Bibliothek von Hugging Face.

## Lernziele

Am Ende dieses Repositorys kannst du:

- Rohtext mit NLTK durch Tokenisierung, Stemming und Lemmatisierung vorverarbeiten.
- einen Spamklassifikator mit scikit-learn erstellen und dabei Bag-of-Words- und TF-IDF-Features verwenden.
- vortrainierte Transformer-Pipelines anwenden, um Text ohne aufgabenspezifisches Training zu klassifizieren (Zero-Shot).
- die passende Transformer-Pipeline und das passende Modell für eine bestimmte NLP-Aufgabe auswählen.
- Word Embeddings mit einem neuronalen Netz erstellen und die gelernten Vektoren betrachten.

## Lernpfad

Bearbeite die Notebooks der Reihe nach. Sie bauen aufeinander auf und führen vom klassischen Machine Learning zu Transformer-basierten Ansätzen.

| Datei / Ordner | Beschreibung |
|---|---|
| [**1 - Spamklassifikator**](1_spam_classifier.ipynb) | SMS-Texte vorverarbeiten und mit scikit-learn einen Spamklassifikator mit logistischer Regression trainieren. |
| [**2 - Zero-Shot für Spam**](2_spam_zero_shot.ipynb) | Eine vortrainierte Transformer-Pipeline verwenden, um Spam per Zero-Shot Learning zu klassifizieren. |
| [**3 - Modelle für Transformer-Pipelines**](3_transformers_zero_shot_pipeline.ipynb) | Untersuchen, wie verschiedene vortrainierte Modelle und Pipelines zu unterschiedlichen NLP-Aufgaben passen. |
| [**4 - Embeddings erstellen**](4_create_embeddings.ipynb) | Word Embeddings mit einem neuronalen Netz in Keras erstellen und die gelernten Vektoren betrachten. |
| [**5 - Optional**](5_optional.ipynb) | Optionale, praxisnahe Aufgabe: mit TensorFlow Word Embeddings für die Spam/Ham-Daten trainieren und sie im Embedding Projector untersuchen. |

### Weitere Ordner und Dateien

| Datei / Ordner | Beschreibung |
|---|---|
| [**Daten**](data/) | Der Datensatz SMS Spam Collection und eine Beispiel-CSV-Datei, die in den Notebooks verwendet werden. |
| [**Extra**](extra/) | Hilfsfunktionen (`utility.py`), die das Notebook zu den Embeddings verwendet. |
| [**Lösungen**](solutions/) | Musterlösungen. |
| [**pyproject.toml**](pyproject.toml) | Projektkonfiguration und Abhängigkeiten. |
| [**uv.lock**](uv.lock) | Lock-Datei für die Abhängigkeiten. |

## Setup

> [!NOTE]
> In allen Schritten ist Text in spitzen Klammern wie `<repo-name>` ein **Platzhalter**. Ersetze ihn samt den Klammern `< >` durch deinen eigenen Wert. Aus `cd <repo-name>` wird zum Beispiel `cd ds-intro-to-NLP`.

### 1. Repository aus dem Template erstellen

Klicke auf GitHub auf **Use this template**.

Beim Erstellen des Repositorys:

- Lege dich selbst als **Owner** fest
- Wähle einen Namen für das Repository
- Deaktiviere **Include all branches**
- Klicke auf **Create repository**

> [!IMPORTANT]
> Wenn du zu zweit oder in einer Gruppe arbeitest, erledigt nur **eine Person** diesen Schritt.

---

### 2. Collaborators hinzufügen (nur bei Paar- oder Gruppenarbeit)

Wenn du mit Teammitgliedern arbeitest:

1. Öffne das Repository auf GitHub
2. Gehe zu **Settings → Collaborators**
3. Füge deine Teammitglieder als Collaborators hinzu
4. Teile den Link zum Repository mit deinem Team

Die Teammitglieder sollten die Einladung annehmen, bevor sie weitermachen.

---

### 3. Repository klonen

Kopiere die SSH-URL über den Button **Code** auf GitHub und führe dann Folgendes aus:

```bash
git clone <copied-ssh-url>
```

Die kopierte SSH-URL hat die Form `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. In den Projektordner wechseln und Abhängigkeiten installieren

Das installiert alle Abhängigkeiten und erstellt eine virtuelle Umgebung in `.venv/`.

```bash
cd <repo-name>
uv sync
```

> [!NOTE]
> Beim ersten Ausführen werden große vortrainierte Modelle von Hugging Face heruntergeladen (Notebook 3 lädt ein Modell von etwa 1,6 GB herunter), plane dafür also Zeit und Speicherplatz ein. Nach dem ersten Herunterladen werden die Modelle lokal zwischengespeichert.

---

### 5. Notebooks öffnen

> [!NOTE]
> Achte darauf, VS Code im Stammverzeichnis des Projekts zu öffnen, damit es die von `uv sync` erstellte Umgebung automatisch erkennt.

Starte VS Code im Stammverzeichnis des Projekts:

```bash
code .
```

Öffne dann ein Notebook und wähle die von `uv sync` erstellte Python-Umgebung als Kernel aus.

## Quellen & weiterführende Literatur

- [**Natural Language Processing with Python (the NLTK book)**](https://www.nltk.org/book/): Die Standardeinführung in die Textverarbeitung mit NLTK
- [**scikit-learn: Text feature extraction**](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction): Offizieller Leitfaden zu CountVectorizer und TF-IDF
- [**Hugging Face: Pipelines**](https://huggingface.co/docs/transformers/main_classes/pipelines): Referenz zur API für Transformer-Pipelines, die in den Notebooks 2 und 3 verwendet wird
- [**Hugging Face: Zero-Shot Classification**](https://huggingface.co/tasks/zero-shot-classification): Was die Zero-Shot-Klassifikation ist und welche Modelle du dafür verwenden kannst
- [**The Illustrated Word2vec**](https://jalammar.github.io/illustrated-word2vec/): Eine visuelle Erklärung, wie Word Embeddings gelernt werden und was sie repräsentieren
