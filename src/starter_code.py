# starter_code.py
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import gensim.downloader as api
import time
import traceback
import pickle

# Load Gensim Word2Vec model globally to avoid repeated loading
try:
    with open('model.dat', 'rb') as f:
        print('Loading Word2Vec model from file...')
        gnews = pickle.load(f)
except FileNotFoundError:
    print('Downloading Word2Vec model...')
    gnews = api.load('word2vec-google-news-300')
    with open('model.dat', 'wb') as f:
        pickle.dump(gnews, f)


def get_word_embeddings(words):
    """Convert words to vector embeddings using Gensim, filtering out OOV words."""
    embeddings = []
    valid_words = []
    
    for word in words:
        if word in gnews:
            embeddings.append(gnews[word])
            valid_words.append(word)
    
    return np.array(embeddings), valid_words


def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    start_time = time.time()
    try:
        if isinstance(words, str):
            try:
                words = ast.literal_eval(words)
                print("Parsed words list:", words)
            except (ValueError, SyntaxError) as e:
                print(f"Parsing error: {e}")
                return ["error", "handling", "default", "guess"], True

        if not (isinstance(words, list) and len(words) == 16):
            print("Invalid input: Returning default guess.")
            return ["error", "handling", "default", "guess"], True

        # Generate embeddings
        word_embeddings, filtered_words = get_word_embeddings(words)
        print("Filtered words:", filtered_words)

        if len(filtered_words) < 8:
            print("Insufficient valid embeddings: Returning default guess.")
            return ["error", "handling", "default", "guess"], True

        print("Word embeddings generated, shape:", word_embeddings.shape)

        # Clustering step with KMeans
        print("Clustering words using KMeans...")
        try:
            n_clusters = min(4, len(filtered_words) // 2)  # Adjust number of clusters if fewer words are available
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(word_embeddings)
            print("KMeans clustering successful.")
        except Exception as e:
            print("Clustering error:", e)
            return words[:4], True  # Return first 4 words to avoid default guess

        # Generate clusters from KMeans labels
        clusters = {i: [] for i in range(n_clusters)}
        for word, label in zip(filtered_words, cluster_labels):
            clusters[label].append(word)

        # Filter clusters of length 4 that have not been guessed before
        cluster_list = [cluster for cluster in clusters.values() if len(cluster) == 4 and cluster not in previousGuesses]
        print("Generated cluster:", cluster_list)

        # Fallback logic: if no valid cluster of size 4, use similarity-based selection
        if not cluster_list:
            print("No valid clusters found, using similarity-based fallback...")
            similarities = cosine_similarity(word_embeddings)
            avg_similarities = similarities.mean(axis=1)
            sorted_indices = np.argsort(avg_similarities)[::-1]
            top_indices = sorted_indices[:4]
            guess = [filtered_words[i] for i in top_indices]
        else:
            guess = cluster_list[0]

        print("Generated guess:", guess)

        # Logic for handling 'One Away' condition
        if isOneAway:
            print("One away condition detected, adjusting guess...")
            if len(correctGroups) > 0:
                correct_group = correctGroups[-1]
                for word in guess:
                    if word not in correct_group:
                        guess.remove(word)
                        break
                guess.extend([word for word in correct_group if word not in guess])
                print("Adjusted guess for one away:", guess)

        endTurn = strikes >= 4 or len(correctGroups) == 4
        return guess, endTurn
    except Exception as e:
        print("Unexpected error in model function:", e)
        traceback.print_exc()
        return ["error", "handling", "default", "guess"], True
    finally:
        end_time = time.time()
        print(f"Model function execution time: {end_time - start_time:.2f} seconds")
