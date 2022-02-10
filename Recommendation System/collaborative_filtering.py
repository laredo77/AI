# Itamar Laredo, 311547087
import math
import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances

class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.mean_user_rating = []
        self.u_ratings_diff, self.i_ratings_diff = [], []
        self.ub_mat, self.ib_mat = [], []
        self.u_prediction_matrix, self.i_prediction_matrix = [], []
        self.userId_to_index, self.index_to_userId = {}, {}
        self.movieId_to_index, self.index_to_movieId = {}, {}
        self.movieId_to_title, self.title_to_movieId = {}, {}

    def create_fake_user(self, rating):
        # my fake user loves drama movies.
        fake_user = {"userId": [283238, 283238, 283238, 283238, 283238],
                     "movieId": [14, 147, 954, 47099, 8958], "rating": [4, 4, 5, 5, 5]}
        fu_df = pd.DataFrame(fake_user)
        rating = pd.concat([rating, fu_df], ignore_index=True)
        return rating

    def initialize_dictionaries(self, data):
        self.userId_to_index = {key: value for value, key in enumerate(self.u_ratings_diff.index)}
        self.index_to_userId = dict(enumerate(self.u_ratings_diff.index))
        self.movieId_to_index = {key: value for value, key in enumerate(self.i_ratings_diff.index)}
        self.index_to_movieId = dict(enumerate(self.i_ratings_diff.index))
        self.movieId_to_title = dict(zip(data[1]['movieId'], data[1]['title']))
        self.title_to_movieId = dict(zip(data[1]['title'], data[1]['movieId']))

    def create_user_based_matrix(self, data):
        ratings = data[0]
        ratings = self.create_fake_user(ratings)
        self.ub_mat = ratings.pivot(index='userId', columns='movieId', values='rating')
        self.mean_user_rating = self.ub_mat.mean(axis=1).to_numpy().reshape(-1, 1)
        self.u_ratings_diff = (self.ub_mat - self.mean_user_rating)
        self.u_ratings_diff[np.isnan(self.u_ratings_diff)] = 0
        self.i_ratings_diff = self.u_ratings_diff.T
        self.initialize_dictionaries(data)
        self.user_based_matrix = 1 - pairwise_distances(self.u_ratings_diff, metric='cosine')
        self.u_prediction_matrix = self.mean_user_rating + (self.user_based_matrix.dot(self.u_ratings_diff) / np.array(
            [np.abs(self.user_based_matrix).sum(axis=1)]).T)

        predictions = self.predict_movies(283238, 5)
        print("My fake user predictions: ", predictions)

    def create_item_based_matrix(self, data):
        ratings = data[0]
        self.ib_mat = ratings.pivot(index='movieId', columns='userId', values='rating')
        self.mean_user_rating = self.ib_mat.T.mean(axis=1).to_numpy().reshape(-1, 1)
        self.i_ratings_diff = (self.ib_mat - self.mean_user_rating.T)
        self.i_ratings_diff[np.isnan(self.i_ratings_diff)] = 0
        self.u_ratings_diff = self.i_ratings_diff.T
        self.initialize_dictionaries(data)
        self.item_based_metrix = 1 - pairwise_distances(self.i_ratings_diff, metric='cosine')
        self.i_prediction_matrix = self.mean_user_rating + (self.item_based_metrix.dot(self.i_ratings_diff) / np.array(
            [np.abs(self.item_based_metrix).sum(axis=1)]).T).T

    def predict_movies(self, user_id, k, is_user_based=True):
        if is_user_based:
            pred = self.u_prediction_matrix
        else:
            pred = self.i_prediction_matrix
        index = self.userId_to_index[int(user_id)]
        user_prediction = pred[index]
        tmp_user_rated = np.where(self.u_ratings_diff.loc[[int(user_id)]].values != 0)[1]
        user_prediction[tmp_user_rated] = 0
        top_k = np.argpartition(user_prediction, -k)[-k:].tolist()
        top_k = sorted(user_prediction[top_k], reverse=True)
        temp_list,  top_k_movies = [], []
        for value in top_k:
            if math.isnan(value):  # handling cold start
                continue
            temp_list.append(list(user_prediction).index(value))
        for i in temp_list:
            movieId = self.index_to_movieId[i]
            title = self.movieId_to_title[movieId]
            top_k_movies.append(title)
        return top_k_movies
