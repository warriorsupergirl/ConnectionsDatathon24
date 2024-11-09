from gensim.models import Word2Vec
import os

# Step 1: Prepare your dataset
# Assume we have a text corpus in 'corpus.txt', where each line is a sentence.

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip().lower().split()

# Path to your corpus file
corpus_file = 'corpus.txt'  # Replace this with your corpus file

# Create a list of tokenized sentences
sentences = list(read_corpus(corpus_file))

# Step 2: Train Word2Vec model
model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4, sg=1)

# vector_size: The dimensionality of the feature vectors.
# window: Maximum distance between the current and predicted word within a sentence.
# min_count: Ignores all words with a total frequency lower than this.
# workers: Number of worker threads to use.
# sg: 1 for skip-gram; 0 for CBOW.

# Step 3: Save the trained model
model.save("../../custom_word2vec.model")

print("Training complete. Model saved as 'custom_word2vec.model'")
