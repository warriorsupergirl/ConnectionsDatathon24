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


# flask --app src/run.py run --host=127.0.0.1 --port=5000
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import spacy
import ast

# Load spaCy model (this can be done once globally to avoid repeated loading)
nlp = spacy.load('en_core_web_md')

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    _______________________________________________________
    Parameters:
    words - 1D Array with 16 shuffled words
    strikes - Integer with number of strikes
    isOneAway - Boolean if your previous guess is one word away from the correct answer
    correctGroups - 2D Array with groups previously guessed correctly
    previousGuesses - 2D Array with previous guesses
    error - String with error message (0 if no error)

    Returns:
    guess - 1D Array with 4 words
    endTurn - Boolean if you want to end the puzzle
    _______________________________________________________
    """

    # Debugging output to verify structure of 'words' input
    print(f"Words (raw): {words}")
    print(f"Type of words: {type(words)}")

    # If 'words' is a string, attempt to parse it as a list
    if isinstance(words, str):
        try:
            # Convert the string representation of the list to an actual list
            words = ast.literal_eval(words)
        except (ValueError, SyntaxError) as e:
            print(f"Parsing error: {e}")
            return ["error", "handling", "default", "guess"], True

    # Ensure 'words' is now a list of exactly 16 elements
    if isinstance(words, list) and len(words) == 16:
        print("Validation passed: words is a list of 16 elements.")
    else:
        print("Invalid input detected: Returning default guess.")
        return ["error", "handling", "default", "guess"], True

    # Example logic to select the first four words as a guess
    guess = words[:4]

    # Logic for endTurn
    endTurn = strikes >= 4

    return guess, endTurn
