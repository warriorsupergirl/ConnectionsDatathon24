import requests
import os
import time

api_key = 'BxMtCMHZgMkFRb3kCgrzgQLBAGeWRaEA'
url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'

# Categories to search
categories = ['technology', 'science', 'health', 'business', 'politics']

articles = []
for category in categories:
    for page in range(5):  # Fetch 5 pages per category, adjust as needed
        params = {
            'q': category,
            'begin_date': '20220101',  # Start date (YYYYMMDD)
            'end_date': '20241110',    # End date (YYYYMMDD)
            'api-key': api_key,
            'page': page
        }

        response = requests.get(url, params=params)

        # Check the response status
        if response.status_code == 200:
            data = response.json()
            print(f"Processing category '{category}', page {page}...")  # Print to verify the category and page being processed

            # Get articles from the response
            new_articles = data.get('response', {}).get('docs', [])
            
            if new_articles:
                articles.extend(new_articles)
            else:
                break  # No more articles, stop pagination
        elif response.status_code == 429:
            print(f"Rate limit exceeded for category '{category}', page {page}. Waiting before retrying...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            continue  # Retry the same page
        else:
            print(f"Failed to retrieve category '{category}', page {page}. Status code: {response.status_code}")
            break

# Ensure the target directory exists
output_directory = os.path.abspath('src/custom-model')
os.makedirs(output_directory, exist_ok=True)

# Set the full output file path
output_file_path = os.path.join(output_directory, 'nyt_corpus.txt')

# Access the articles and save content to a file
if articles:
    with open(output_file_path, 'w+', encoding='utf-8') as file:
        for article in articles:
            headline = article.get('headline', {}).get('main', '')
            snippet = article.get('snippet', '')
            lead_paragraph = article.get('lead_paragraph', '')
            
            # Write each element into the file, separating with newline
            if headline:
                file.write(headline + '\n')
            if snippet:
                file.write(snippet + '\n')
            if lead_paragraph:
                file.write(lead_paragraph + '\n')
            file.write('\n')  # Separate articles with a blank line

    print("Articles saved to nyt_corpus.txt.")
else:
    print("No articles found for the given date range.")
