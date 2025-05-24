import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset of movies and user ratings
# In real life, you would load your own data
np.random.seed(42)  # for reproducibility

# Create sample movie data
movies = {
    'movie_id': range(1, 11),
    'title': [
        'The Matrix', 'Inception', 'Titanic', 'Avatar', 'Jurassic Park',
        'Star Wars', 'The Godfather', 'Forrest Gump', 'The Dark Knight', 'Pulp Fiction'
    ],
    'action': [9, 8, 2, 7, 8, 8, 5, 3, 9, 7],  # Action score (0-10)
    'romance': [3, 4, 9, 5, 2, 3, 4, 8, 2, 3],  # Romance score (0-10)
    'comedy': [2, 3, 4, 3, 5, 4, 2, 7, 3, 8]   # Comedy score (0-10)
}

# Create DataFrame
movies_df = pd.DataFrame(movies)

# Step 2: Prepare features for the model
features = ['action', 'romance', 'comedy']
X = movies_df[features]

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Create and train the KNN model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_scaled)

# Step 5: Function to get movie recommendations
def get_movie_recommendations(movie_title):
    # Find the movie index
    movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Get the movie's features
    movie_features = X_scaled[movie_idx].reshape(1, -1)
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors(movie_features)
    
    # Get recommended movies
    recommended_movies = movies_df.iloc[indices[0]]
    
    return recommended_movies

# Step 6: Visualize the movie space
def show_visualization():
    plt.figure(figsize=(10, 6))
    plt.scatter(X['action'], X['romance'], c=X['comedy'], cmap='viridis')
    for i, title in enumerate(movies_df['title']):
        plt.annotate(title, (X['action'][i], X['romance'][i]))
    plt.xlabel('Action Score')
    plt.ylabel('Romance Score')
    plt.title('Movie Space Visualization\n(Color represents Comedy Score)')
    plt.colorbar(label='Comedy Score')
    plt.show(block=False)  # Changed to non-blocking
    plt.pause(0.1)  # Added small pause

# Step 7: Get recommendations for a sample movie
sample_movie = 'The Matrix'
print(f"\nRecommendations for '{sample_movie}':")
recommendations = get_movie_recommendations(sample_movie)
print("\nRecommended movies:")
for _, movie in recommendations.iterrows():
    print(f"\nTitle: {movie['title']}")
    print(f"Action: {movie['action']}/10")
    print(f"Romance: {movie['romance']}/10")
    print(f"Comedy: {movie['comedy']}/10")

# Step 8: Interactive recommendation system
def recommend_movies():
    print("\nAvailable movies:")
    for title in movies_df['title']:
        print(f"- {title}")
    
    while True:
        movie = input("\nEnter a movie title (or 'quit' to exit): ")
        if movie.lower() == 'quit':
            plt.close('all')  # Close all plots
            break
        if movie in movies_df['title'].values:
            recommendations = get_movie_recommendations(movie)
            print(f"\nRecommendations for '{movie}':")
            for _, rec in recommendations.iterrows():
                print(f"- {rec['title']}")
        else:
            print("Movie not found. Please try again.")

# Show visualization and run the interactive system
if __name__ == "__main__":
    show_visualization()
    recommend_movies()
