# Itamar Laredo
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def precision_10(test_set, cf, is_user_based=True):
    u_dictionary = {}
    hits, k, min_rate = 0, 10, 4
    movieId_prediction_list = []
    for u_id, m_id, rating in zip(test_set.userId, test_set.movieId, test_set.rating):
        if rating >= min_rate:
            if u_id in u_dictionary.keys():
                u_dictionary[u_id].append(m_id)
            else:
                u_dictionary[u_id] = [m_id]

    for user in u_dictionary.keys():
        user_rating_list = u_dictionary[user]
        user_pred_list_titled = cf.predict_movies(user, k, is_user_based)
        for title in user_pred_list_titled:
            movieId_prediction_list.append(cf.title_to_movieId[title])
        intersection = list(set(user_rating_list) & set(movieId_prediction_list))
        hits += len(intersection)
        movieId_prediction_list.clear()
    val = (hits / k) / len(u_dictionary.keys())
    print("Precision_k: " + str(val))

def ARHA(test_set, cf, is_user_based=True):
    u_dictionary = {}
    arhr_sum, k, min_rate = 0, 10, 4
    movieId_prediction_list = []
    for u_id, m_id, rating in zip(test_set.userId, test_set.movieId, test_set.rating):
        if rating >= min_rate:
            if u_id in u_dictionary.keys():
                u_dictionary[u_id].append(m_id)
            else:
                u_dictionary[u_id] = [m_id]

    for user in u_dictionary.keys():
        user_rating_list = u_dictionary[user]
        user_pred_list_titled = cf.predict_movies(user, k, is_user_based)
        for title in user_pred_list_titled:
            movieId_prediction_list.append(cf.title_to_movieId[title])
        intersection = list(set(user_rating_list) & set(movieId_prediction_list))
        for i in intersection:
            arhr_sum += (1 / (movieId_prediction_list.index(i) + 1))
        movieId_prediction_list.clear()

    val = arhr_sum / len(u_dictionary.keys())
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based=True):
    if is_user_based:
        prediction_matrix = cf.u_prediction_matrix
    else:
        prediction_matrix = cf.i_prediction_matrix

    rsme_numerator = 0
    for u_id, m_id, rating in zip(test_set.userId, test_set.movieId, test_set.rating):
        i, j = cf.userId_to_index[u_id], cf.movieId_to_index[m_id]
        prediction_rating = prediction_matrix[i][j]
        precision = (prediction_rating - rating)**2
        if math.isnan(precision):
            continue
        rsme_numerator += precision

    val = sqrt(rsme_numerator / len(test_set))
    print("RMSE: " + str(val))
