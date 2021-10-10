import numpy as np
import pandas
from landmark_calculation import calc_landmark


def get_actual_distance_matrix(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


def get_distance_matrix_euclidean(input_csv_file):
    df_euclidean = pandas.read_csv('Euclidean/' + input_csv_file + '.csv', header=None)
    euclidean_mat = df_euclidean.values

    return euclidean_mat


def get_distance_matrix_landmark(input_csv_file, no_landmarks):
    actual_distance = get_actual_distance_matrix(input_csv_file)
    return calc_landmark(actual_distance, no_landmarks, input_csv_file)


def get_predicted_distance_matrix_DNN_K(input_csv_file):
    df_dnn = pandas.read_csv('Outputs_K/' + input_csv_file + '.csv', header=None)
    dnn_mat = df_dnn.values
    return dnn_mat


# euclidean_distance = get_distance_matrix_euclidean(map)
# landmark_distance = get_distance_matrix_landmark('brc300d', 100)
# predicted_distance_matrix_dnn_K = get_predicted_distance_matrix_DNN_K(map)
#
# # adjacency_matrix = (np.abs(actual_distance - euclidean_distance) == 0)
# # print(adjacency_matrix)
#
# adjacency_matrix = np.abs(actual_distance - euclidean_distance)
#
# print(np.mean(adjacency_matrix))
# print(np.std(adjacency_matrix))
# print("\n\n")
#
# adjacency_matrix = np.abs(actual_distance - landmark_distance)
#
# print(np.mean(adjacency_matrix))
# print(np.std(adjacency_matrix))
# print("\n\n")
#
# adjacency_matrix = np.abs(actual_distance - np.maximum(euclidean_distance, landmark_distance))
#
# print(np.mean(adjacency_matrix))
# print(np.std(adjacency_matrix))
#
# adjacency_matrix = np.abs(actual_distance - predicted_distance_matrix_dnn_K)
#
# print(np.mean(adjacency_matrix))
# print(np.std(adjacency_matrix))
# print("\n\n")
#
