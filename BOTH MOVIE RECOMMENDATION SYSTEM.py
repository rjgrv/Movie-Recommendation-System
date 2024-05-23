import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

# Load data
movies_data_indian = pd.read_csv(r"C:\Users\Raj Gaurav\Downloads\indian movies.csv")
movies_data = pd.read_csv(r"C:\Users\Raj Gaurav\Downloads\movies.csv")

# Process Indian movies data
selected_features_indian = ['Movie Name', 'Year', 'Rating(10)', 'Genre', 'Language']
for feature in selected_features_indian:
    movies_data_indian[feature] = movies_data_indian[feature].fillna('')

combined_features_indian = movies_data_indian['Movie Name'] + ' ' + movies_data_indian['Year'] + ' ' + movies_data_indian['Rating(10)'] + ' ' + movies_data_indian['Genre'] + ' ' + movies_data_indian['Language']

# Process other movies data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Combine features from both datasets
all_combined_features = pd.concat([combined_features_indian, combined_features], ignore_index=True)

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(all_combined_features)

# Split the feature vectors back to their respective datasets
feature_vectors_indian = feature_vectors[:len(movies_data_indian)]
feature_vectors_other = feature_vectors[len(movies_data_indian):]

# Get user input for favorite movie name
movie_name = input("Enter your favourite movie name: ")

# Combine movie titles for finding close match
all_titles_indian = movies_data_indian['Movie Name'].tolist()
all_titles = movies_data['title'].tolist()
all_titles_combined = all_titles_indian + all_titles

# Find the closest match for the movie name
find_close_match = difflib.get_close_matches(movie_name, all_titles_combined)

if not find_close_match:
    print("No close matches found. Please try another movie name.")
else:
    close_match = find_close_match[0]

    if close_match in all_titles_indian:
        index_of_the_movie = movies_data_indian[movies_data_indian['Movie Name'] == close_match].index[0]
    else:
        index_of_the_movie = len(movies_data_indian) + movies_data[movies_data['title'] == close_match].index[0]

    # Compute cosine similarity incrementally
    similarity_scores = cosine_similarity(feature_vectors[index_of_the_movie], feature_vectors).flatten()

    # Sort the movies based on similarity scores
    sorted_similar_movies = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)

    # Print the top 30 similar movie recommendations without duplicates
    print('Movies suggested for you: \n')
    printed_titles = set()
    for i, movie in enumerate(sorted_similar_movies[1:], start=1):
        index = movie[0]
        if index < len(movies_data_indian):
            title_from_index = movies_data_indian.loc[index, 'Movie Name']
        else:
            title_from_index = movies_data.loc[index - len(movies_data_indian), 'title']
        
        if title_from_index not in printed_titles:
            print(f"{i}. {title_from_index}")
            printed_titles.add(title_from_index)
        
        if len(printed_titles) > 30:
            break
