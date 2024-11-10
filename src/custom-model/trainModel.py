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
with open('src/custom-model/nyt_corpus.txt', 'r', encoding='utf-8') as file:
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
