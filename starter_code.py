# def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
# 	"""
# 	_______________________________________________________
# 	Parameters:
# 	words - 1D Array with 16 shuffled words
# 	strikes - Integer with number of strikes
# 	isOneAway - Boolean if your previous guess is one word away from the correct answer
# 	correctGroups - 2D Array with groups previously guessed correctly
# 	previousGuesses - 2D Array with previous guesses
# 	error - String with error message (0 if no error)

# 	Returns:
# 	guess - 1D Array with 4 words
# 	endTurn - Boolean if you want to end the puzzle
# 	_______________________________________________________
# 	"""

# 	# Your Code here
# 	# Good Luck!

# 	# Example code where guess is hard-coded
# 	guess = ["apples", "bananas", "oranges", "grapes"] # 1D Array with 4 elements containing guess
# 	endTurn = False # True if you want to end puzzle and skip to the next one

# 	return guess, endTurn

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import spacy

# Load spaCy model (this can be done once globally to avoid repeated loading)
nlp = spacy.load('en_core_web_md')

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    Solves the Connections AI game by grouping similar words based on word embeddings.
    """
    
    # Step 1: Convert words to embeddings
    word_vectors = np.array([nlp(word).vector for word in words])

    # Step 2: Calculate clusters of similar words
    clustering = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='average')
    clusters = clustering.fit_predict(word_vectors)

    # Step 3: Group words by clusters
    grouped_words = [[] for _ in range(4)]
    for i, cluster_id in enumerate(clusters):
        grouped_words[cluster_id].append(words[i])

    # Step 4: Generate guesses
    # Select one of the groups to guess. Iterate through grouped_words to make guesses.
    guess = []
    for group in grouped_words:
        if group not in correctGroups and group not in previousGuesses:
            guess = group
            break
    
    # If no new groups to guess, end the turn
    endTurn = len(guess) == 0 or strikes >= 4

    return guess, endTurn
