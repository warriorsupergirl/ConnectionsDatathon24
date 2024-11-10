# starter_code.py
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import time
import random
import traceback
import pickle
import os

model_path = 'model/model.dat'

# Load Gensim Word2Vec model globally to avoid repeated loading
try:
    if not os.path.exists('model'):
        os.makedirs('model')
    with open(model_path, 'rb') as f:
        print('Loading Word2Vec model from file...')
        gnews = pickle.load(f)
        print('Model loaded successfully.')
except FileNotFoundError:
    print('Downloading Word2Vec model...')
    from gensim.downloader import load
    gnews = load('word2vec-google-news-300')
    with open(model_path, 'wb') as f:
        pickle.dump(gnews, f)
        print('Model downloaded and saved successfully.')
except Exception as e:
    print(f"Error loading model: {e}")
    gnews = None


def get_word_embeddings(words):
    """Convert words to vector embeddings using Gensim, filtering out OOV words."""
    embeddings = []
    valid_words = []
    
    if gnews is None:
        print("Model not loaded. Returning empty embeddings.")
        return np.array(embeddings), valid_words

    for word in words:
        if word in gnews:
            embeddings.append(gnews[word])
            valid_words.append(word)
    
    return np.array(embeddings), valid_words


def cluster_words(word_embeddings, filtered_words, previousGuesses):
    """Cluster words using KMeans and filter valid clusters."""
    try:
        if len(filtered_words) < 4:
            print("Not enough words for clustering.")
            return None
        n_clusters = min(4, len(filtered_words) // 2)  # Adjust number of clusters if fewer words are available
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(word_embeddings)
        print("KMeans clustering successful.")
    except Exception as e:
        print("Clustering error:", e)
        return None  # Indicate clustering failure

    # Generate clusters from KMeans labels
    clusters = {i: [] for i in range(n_clusters)}
    for word, label in zip(filtered_words, cluster_labels):
        clusters[label].append(word)

    # Filter clusters of length 4 that have not been guessed before
    cluster_list = [cluster for cluster in clusters.values() if len(cluster) == 4 and cluster not in previousGuesses]
    print("Generated cluster:", cluster_list)
    return cluster_list


def similarity_fallback(word_embeddings, filtered_words):
    """Fallback logic using similarity-based selection."""
    print("No valid clusters found, using similarity-based fallback...")
    similarities = cosine_similarity(word_embeddings)
    avg_similarities = similarities.mean(axis=1)
    sorted_indices = np.argsort(avg_similarities)[::-1]
    top_indices = sorted_indices[:4]
    return [filtered_words[i] for i in top_indices]


def adjust_for_one_away(correctGroups, filtered_words, guess, previousGuesses):
    """Adjust the guess if the 'One Away' condition is detected."""
    print("One away condition detected, adjusting guess...")
    if len(correctGroups) > 0:
        correct_group = correctGroups[-1]
        new_guess = list(correct_group)  # Start with the correct group
        remaining_words = [word for word in filtered_words if word not in new_guess]
        random.shuffle(remaining_words)

        # Replace words in the guess to make it different
        for i in range(len(new_guess)):
            if new_guess[i] not in guess:
                new_guess[i] = remaining_words.pop() if remaining_words else new_guess[i]

        # Track similarity scores to ensure high cohesiveness
        word_embeddings, _ = get_word_embeddings(new_guess)
        if len(word_embeddings) > 0:
            avg_similarity = cosine_similarity(word_embeddings).mean()
            while new_guess in previousGuesses or avg_similarity < 0.5:  # Ensure similarity and avoid duplicates
                random.shuffle(remaining_words)
                for i in range(len(new_guess)):
                    new_guess[i] = remaining_words.pop() if remaining_words else new_guess[i]
                word_embeddings, _ = get_word_embeddings(new_guess)
                if len(word_embeddings) > 0:
                    avg_similarity = cosine_similarity(word_embeddings).mean()

        print("Adjusted guess for one away:", new_guess)
        return new_guess
    return guess


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

        if len(filtered_words) < 4:
            print("Insufficient valid embeddings: Returning default guess.")
            return ["error", "handling", "default", "guess"], True

        print("Word embeddings generated, shape:", word_embeddings.shape)

        # Clustering step with KMeans
        cluster_list = cluster_words(word_embeddings, filtered_words, previousGuesses)

        # Fallback logic: if no valid cluster of size 4, use similarity-based selection
        if not cluster_list:
            guess = similarity_fallback(word_embeddings, filtered_words)
        else:
            guess = cluster_list[0]

        # Ensure the guess is not a duplicate of a previous guess
        if guess in previousGuesses:
            print("Duplicate guess detected, adjusting...")
            random.shuffle(filtered_words)
            guess = filtered_words[:4]

        print("Generated guess:", guess)

        # Logic for handling 'One Away' condition
        if isOneAway:
            guess = adjust_for_one_away(correctGroups, filtered_words, guess, previousGuesses)

        endTurn = strikes >= 4 or len(correctGroups) == 4
        return guess, endTurn
    except Exception as e:
        print("Unexpected error in model function:", e)
        traceback.print_exc()
        return ["error", "handling", "default", "guess"], True
    finally:
        end_time = time.time()
        print(f"Model function execution time: {end_time - start_time:.2f} seconds")
