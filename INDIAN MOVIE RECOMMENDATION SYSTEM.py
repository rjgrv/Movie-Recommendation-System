#RAJ GAURAV

import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load data
movies_data = pd.read_csv(r"C:\Users\Raj Gaurav\Downloads\indian movies.csv")

# Select relevant features and handle missing values
selected_features = ['Movie Name', 'Year', 'Rating(10)', 'Genre', 'Language']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single string for each movie
combined_features = movies_data['Movie Name'] + ' ' + movies_data['Year'] + ' ' + movies_data['Rating(10)'] + ' ' + movies_data['Genre'] + ' ' + movies_data['Language']

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Convert feature vectors to sparse matrix
sparse_feature_vectors = csr_matrix(feature_vectors)

# Get user input for favorite movie name
movie_name = input("Enter your favourite movie name: ")

# Find the closest match for the movie name
list_of_all_titles = movies_data['Movie Name'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

if not find_close_match:
    print("No close matches found. Please try another movie name.")
else:
    close_match = find_close_match[0]

    # Get index of the matched movie
    index_of_the_movie = movies_data[movies_data['Movie Name'] == close_match].index[0]

    # Compute cosine similarity incrementally
    similarity_scores = cosine_similarity(sparse_feature_vectors[index_of_the_movie], sparse_feature_vectors).flatten()

    # Sort the movies based on similarity scores
    sorted_similar_movies = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)

    # Print the top 30 similar movie recommendations
    print('Movies suggested for you: \n')
    for i, movie in enumerate(sorted_similar_movies[1:31], start=1):
        index = movie[0]
        title_from_index = movies_data.loc[index, 'Movie Name']
        print(f"{i}. {title_from_index}")
