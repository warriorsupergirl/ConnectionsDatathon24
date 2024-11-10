import aiohttp
import asyncio

# Base URL for the ConceptNet API
CONCEPTNET_API_URL = "http://api.conceptnet.io/c/en/"

# Asynchronous function to get related concepts using the related endpoint
async def get_related_concepts(word, session, limit=5):
    url = f"{CONCEPTNET_API_URL}{word}/related"
    
    try:
        async with session.get(url) as response:
            # If the request was successful
            if response.status == 200:
                data = await response.json()
                
                # Print the raw response for debugging
                print(f"Response for '{word}':", data)

                related = []

                # Extract the related words from the response
                for edge in data.get('edges', [])[:limit]:  # Limit the number of related concepts
                    related_word = edge['end']['label']
                    relation = edge['rel']['label']
                    related.append({'word': related_word, 'relation': relation})
                return related
            else:
                print(f"Error fetching data for {word}: {response.status}")
                return []
    except Exception as e:
        print(f"Error occurred for {word}: {e}")
        return []

# Fallback search function if /related fails
async def search_concept(word, session, limit=5):
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}"

    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print(f"Search Response for '{word}':", data)

                related = []
                for edge in data.get('edges', [])[:limit]:
                    related_word = edge['end']['label']
                    relation = edge['rel']['label']
                    related.append({'word': related_word, 'relation': relation})
                return related
            else:
                print(f"Error fetching search data for {word}: {response.status}")
                return []
    except Exception as e:
        print(f"Error occurred during search for {word}: {e}")
        return []

# Asynchronous function to handle multiple requests with fallback logic
async def get_multiple_related_concepts(words, limit=5):
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Create a task for each word, first try /related, then fallback to /search
        for word in words:
            tasks.append(get_related_concepts(word, session, limit))
        
        # Wait for all tasks to finish
        results = await asyncio.gather(*tasks)

        # If any results are empty, fallback to /search
        for i, result in enumerate(results):
            if not result:
                print(f"Trying search for '{words[i]}'.")
                results[i] = await search_concept(words[i], session, limit)

        return results

# Example usage: Get related concepts for multiple words
words_to_query = ['dog', 'cat', 'apple', 'car']
limit = 5

# Run the asynchronous query
async def main():
    related_words = await get_multiple_related_concepts(words_to_query, limit)
    
    # Print the results for each word
    for word, related in zip(words_to_query, related_words):
        print(f"Related words for '{word}':")
        if related:  # Only print if there are related words
            for item in related:
                print(f"Related Word: {item['word']} | Relation: {item['relation']}")
        else:
            print("No related words found.")
        print()

# Run the main function
asyncio.run(main())
