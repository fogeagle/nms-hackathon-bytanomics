# #internship - supplemental income
# import googlesearch as gs
# import requests as r
# url = 'https://api.indeed.com/search?='
# search = input("Enter some job pls: ")
# def searchurlconv(search):
#     pass
# response = r.get(url)#searchurlconv(search))
# # x = gs.search("Google",num_results=10)
# # for i in x:
# #     print(i)


import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Step 1: Data Collection
def fetch_job_listings():
    indeed_api_key = "YOUR_INDEED_API_KEY"
    search_params = {
        'publisher': indeed_api_key,
        'q': 'python developer',
        'l': 'remote',  # Example location filter
        'limit': 50  # Example limit
    }
    response = requests.get('https://api.indeed.com/ads/apisearch', params=search_params)
    job_listings = response.json()['results']
    return job_listings

# Step 2: Data Preprocessing
def preprocess_job_listings(job_listings):
    df = pd.DataFrame(job_listings)
    # Perform data cleaning, text standardization, etc.
    return df

# Step 3: Feature Engineering
def extract_features(df):
    tfidf_vectorizer = TfidfVectorizer()
    job_descriptions_tfidf = tfidf_vectorizer.fit_transform(df['description'])
    return job_descriptions_tfidf

# Step 4: Model Training
def train_model(features):
    model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
    model.fit(features)
    return model

# Step 5: Evaluation (skipping for simplicity)

# Step 6: Deployment (skipping for simplicity)

# Step 7: Continuous Monitoring and Updating (skipping for simplicity)

if __name__ == "__main__":
    # Step 1: Data Collection
    job_listings = fetch_job_listings()

    # Step 2: Data Preprocessing
    df = preprocess_job_listings(job_listings)

    # Step 3: Feature Engineering
    job_features = extract_features(df)

    # Step 4: Model Training
    model = train_model(job_features)

    # Example query to find similar jobs
    query = "python developer position with experience in machine learning"
    query_tfidf = tfidf_vectorizer.transform([query])
    distances, indices = model.kneighbors(query_tfidf, n_neighbors=5)

    # Print similar job titles
    similar_jobs = df.iloc[indices[0]]['jobtitle']
    print(similar_jobs)
