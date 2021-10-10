import csv
import numpy as np
import pandas


# find landmarks

def get_actual_distance_matrix(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


def get_original_landmark_matrix(input_csv_file, no_landmarks):
    df_land = pandas.read_csv('Landmark/' + input_csv_file + '_' + str(no_landmarks) + '.csv', header=None)
    landmark_mat = df_land.values
    return landmark_mat


def find_landmarks(actual_distance_, no_landmarks_):
    random_node = 0
    landmark_list_ = []
    landmark_id = np.argmax(actual_distance_[random_node])
    landmark_list_.append(landmark_id)

    for i in range(no_landmarks_ - 1):

        max_val = 0
        max_id = 0

        for j in range(len(actual_distance_)):
            curr_d = 0
            for landmark in landmark_list_:
                if landmark == j:
                    curr_d = 0
                    break
                d = actual_distance_[landmark][j]
                curr_d = curr_d + d

            if curr_d > max_val:
                max_val = curr_d
                max_id = j
            elif curr_d < 0:
                max_id = j
                break

        landmark_list_.append(max_id)

    return landmark_list_


def calc_landmark(actual_distance_, no_landmarks_, input_csv_file):
    landmark_list_ = find_landmarks(actual_distance_, no_landmarks_)
    # num_nodes = len(actual_distance)
    landmark_dist = np.zeros((len(actual_distance_), len(actual_distance_)))

    for i in range(len(actual_distance_)):
        for j in range(i + 1, len(actual_distance_)):
            start = i
            end = j
            max_dist = 0
            for k in range(len(landmark_list_)):
                st_dist = actual_distance_[landmark_list_[k]][start]
                en_dist = actual_distance_[landmark_list_[k]][end]
                max_dist = max(max_dist, np.abs(st_dist - en_dist))

            landmark_dist[i][j] = landmark_dist[j][i] = max_dist

    # print(np.sum((landmark_dist - landmark_original) ** 2) / num_nodes ** 2)
    # print(np.sum((landmark_dist - actual_distance) ** 2) / num_nodes ** 2)
    # print(np.sum((landmark_original - actual_distance) ** 2) / num_nodes ** 2)

    with open("Landmark_makeshift/" + input_csv_file + "_" + str(no_landmarks_) + ".csv", mode='w',
              newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in landmark_dist:
            csv_file_writer.writerow(row)

    with open("Landmark_makeshift_list/" + input_csv_file + "_" + str(no_landmarks_) + ".csv", mode='w',
              newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_file_writer.writerow(landmark_list_)

    return landmark_dist

# calc_landmark(actual_distance, no_landmarks, map)
