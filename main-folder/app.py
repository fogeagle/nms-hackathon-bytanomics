from flask import Flask, render_template, request, jsonify
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

class Job:
    def __init__(self, title, description):
        self.title = title
        self.description = description

def fetch_jobs_from_api():
    # Example: Using GitHub Jobs API
    response = requests.get("https://jobs.github.com/positions.json")
    jobs_data = response.json()
    jobs = []
    for job_data in jobs_data:
        job = Job(job_data['title'], job_data['description'])
        jobs.append(job)
    return jobs

def match_people_to_jobs(people, jobs):
    # Vectorize job descriptions
    job_descriptions = [job.description for job in jobs]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(job_descriptions)

    matches = []
    for person in people:
        # Vectorize person's skills
        person_skills = ' '.join(person.skills)
        Y = vectorizer.transform([person_skills])

        # Calculate cosine similarity between person's skills and job descriptions
        similarities = cosine_similarity(X, Y)

        # Find the best match
        best_match_index = similarities.argmax()
        best_match_score = similarities[best_match_index]

        # If the best match score is above a certain threshold, consider it a match
        if best_match_score > 0.5:
            best_match_job = jobs[best_match_index]
            matches.append((person.name, best_match_job.title, best_match_job.description))

    return matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_jobs', methods=['POST'])
def find_jobs():
    name = request.form['name']
    skills = request.form['skills'].split(',')

    # Sample people (you can replace these with actual data)
    people = [
        {"name": "Alice", "skills": ["Python", "Java", "SQL"]},
        {"name": "Bob", "skills": ["Java", "C++", "HTML"]},
        {"name": "Charlie", "skills": ["Python", "JavaScript", "CSS"]}
    ]

    jobs = fetch_jobs_from_api()

    matched_jobs = match_people_to_jobs(people, jobs)
    
    return jsonify(matched_jobs)

if __name__ == '__main__':
    app.run(debug=True)