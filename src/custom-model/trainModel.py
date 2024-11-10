import re
import spacy
from gensim.models import Word2Vec
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Enable logging to track the progress of training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Load and preprocess the data
with open('./nyt_corpus.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Lowercase the text
raw_text = raw_text.lower()

# Remove punctuation except sentence-ending punctuation and other non-word characters
raw_text = re.sub(r'[^a-zA-Z0-9\s\.]', '', raw_text)

# Use spaCy to process the text
doc = nlp(raw_text)

# Split into sentences and tokenize using spaCy
filtered_sentences = []
for sent in doc.sents:
    filtered_sentence = [token.text for token in sent if not token.is_stop and not token.is_punct]
    filtered_sentences.append(filtered_sentence)

# Set parameters for the Word2Vec model
vector_size = 200
window_size = 7
min_word_count = 1
num_workers = 4
skip_gram = 1  # Use skip-gram model (1) or CBOW (0)

# Train Word2Vec model
model = Word2Vec(
    sentences=filtered_sentences,
    vector_size=vector_size,
    window=window_size,
    min_count=min_word_count,
    workers=num_workers,
    sg=skip_gram
)

# Save the trained model
model.save("nyt_word2vec.model")
print("Word2Vec model training complete and saved to nyt_word2vec.model.")

# Visualization of the word embeddings using PCA
def visualize_embeddings(model, words_to_visualize=20):
    words = list(model.wv.index_to_key)[:words_to_visualize]
    word_vectors = [model.wv[word] for word in words]

    # Reduce the dimensionality of word vectors to 2D
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    # Plot the words and their embeddings
    plt.figure(figsize=(7, 7))
    for word, coord in zip(words, reduced_vectors):
        plt.scatter(coord[0], coord[1])
        plt.text(coord[0] + 0.02, coord[1] + 0.02, word, fontsize=9)
    
    plt.title("Word Embeddings Visualized (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()

# Visualize the embeddings for the top 20 words
visualize_embeddings(model)

import requests
from langdetect import detect

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def get_related_terms(word, language='en', limit=100):
    url = f'http://api.conceptnet.io/c/{language}/{word}?offset=0&limit={limit}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        data = response.json()
        related_terms = set()  # Using a set to prevent duplicates

        for edge in data.get('edges', []):
            start = edge['start']['label']
            end = edge['end']['label']
            if start.lower() != word.lower() :
                related_terms.add(start)
            if end.lower() != word.lower() :
                related_terms.add(end)

        return list(related_terms)

    except requests.exceptions.RequestException as e:
        print(f"Request failed for word '{word}': {e}")
        return []

# Build a training corpus
def build_corpus(words, language='en'):
    corpus = []
    for i, word in enumerate(words):
        print(f"Fetching related terms for '{word}' ({i + 1}/{len(words)})...")
        related_terms = get_related_terms(word, language)
        if related_terms:
            corpus.append(related_terms)
        #time.sleep(1)  # Prevents overwhelming the API
    return corpus

# Seed words for building the corpus
seed_words = ['cat', 'dog', 'car', 'apple']
def isen(word):
    for c in word:
        if c.lower() not in "abcdefghijklmnopqrstuvwxyz":
            return False
    return True
# Build the corpus
new_corpus = build_corpus(seed_words)
new_corpus = [[word for word in sentence if isen(word)] for sentence in new_corpus]
# Load or initialize Word2Vec model
model = Word2Vec(vector_size=100, min_count=1)

# # Update the model's vocabulary with the new data
model.build_vocab(new_corpus)#, update=True)

# Train the model further with the new corpus
model.train(new_corpus, total_examples=len(new_corpus), epochs=model.epochs)

# Save the model to a file
model.save('conceptnet_word2vec.model')

# Optional: Test the model
print(model.wv.most_similar('cat'))

# Save related terms to a text file
with open('related_terms.txt', 'w+', encoding='utf-8') as file:
    for sentence in new_corpus:
        file.write(' '.join(sentence) + '\n')

# Visualization (ensure this function is implemented)
visualize_embeddings(model)