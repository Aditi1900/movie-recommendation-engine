import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# helper functions. Use them when needed #######
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


# Here df stands for DataFrame

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


##################################################

# Step 1: Read CSV File

# This function takes the path to the csv file as an argument and returns a pandas DataFrame
df = pd.read_csv('movie_dataset.csv', low_memory=False)

# To display the first few rows of the DataFrame
# print(df.head())

# print(df.columns)

# Step 2: Select Features
features = ['keywords', 'cast', 'genres', 'director']

# Step 3: Create a column in DF which combines all selected features

# This loop will iterate through all rows for the 4 features mentioned above and fill all the NaNs with a black space
for feature in features:
    df[feature] = df[feature].fillna('')

# Here we are creating a function that combines all these cols into one string
def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
    except:
        print("Error:", row)

# axis 1 passes each row individually and not cols
df["combined_features"] = df.apply(combine_features, axis=1)

# print("Combined Features:", df["combined_features"].head())

# Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
count_matrix.toarray()

# Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

movie_user_likes = "Avatar"

# Step 6: Get index of this movie from its title
# We are just calling the helper function mentioned on the top
movie_index = int(get_index_from_title(movie_user_likes))

# we will get the similarity scores of all movies with this specified movie index and store then as a list first and
# then we will enumerate it to get a list of tuples as we have to sort it in descending order after this
similar_movies = list(enumerate(cosine_sim[movie_index]))

# Step 7: Get a list of similar movies in descending order of similarity score
# The lamda function gets the second element of the tuple
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# Step 8: Print titles of first 50 movies
i = 0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i = i+1
    if i > 50:
        break
