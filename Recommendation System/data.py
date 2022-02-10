# Itamar Laredo, 311547087
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):

    users = data[0]["userId"].tolist()
    unique_users = list(set(users))
    print("Unique users ranking: ", len(unique_users))
    movies = data[0]["movieId"].tolist()
    unique_movies = list(set(movies))
    print("Unique movies ranked: ", len(unique_movies))
    rating = data[0]["rating"].tolist()
    print("Amount of rankings: ", len(rating))
    count = Counter(movies)
    most_movie_rated = count.most_common(1)[0][1]
    print("Maximum rankings for a movie: ", most_movie_rated)
    least_movie_rated = count.most_common()[-1][1]
    print("Minimum rankings for a movie: ", least_movie_rated)
    count = Counter(users)
    most_user_rate = count.most_common(1)[0][1]
    print("Maximum rankings for a user: ", most_user_rate)
    least_user_rate = count.most_common()[-1][1]
    print("Minimum rankings for a user: ", least_user_rate)

def plot_data(data, plot = True):
    rating = data[0]["rating"].tolist()
    sorted_rating = rating
    sorted_rating = sorted(list(sorted_rating))
    frequency = Counter(sorted_rating).values()
    sorted_rating = set(sorted_rating)
    sorted_rating = sorted(list(sorted_rating))
    plt.bar(sorted_rating, frequency)
    plt.xlabel("rating")
    plt.ylabel("amount of rates")
    plt.title('Rating Distribution')
    plt.savefig('./plot/graph.png')

