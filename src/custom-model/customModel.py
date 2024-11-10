from gensim.models import Word2Vec

# Load your trained model
custom_model = Word2Vec.load("custom_word2vec.model")

# Example usage
vector = custom_model.wv['example']  # Get vector for a word
similar_words = custom_model.wv.most_similar('example')  # Find similar words

print("Vector for 'example':", vector)
print("Most similar words to 'example':", similar_words)