from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import pandas

map = 'brc200d'
no_landmarks_default = 12


def get_points(input_csv_file):
    df = pandas.read_csv('Coordinates/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


def get_actual_distance_matrix(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


def get_predicted_distance_matrix_landmark(input_csv_file, no_landmarks):
    df_land = pandas.read_csv('Landmark_makeshift/' + input_csv_file + '_' + str(no_landmarks) + '.csv', header=None)
    landmark_mat = df_land.values

    df_land_ids = pandas.read_csv('Landmark_makeshift_list/' + input_csv_file + '_' + str(no_landmarks) + '.csv',
                                  header=None)
    land_ids = df_land_ids.values
    return land_ids, landmark_mat


def get_predicted_distance_matrix_euclidean(input_csv_file):
    df_euclidean = pandas.read_csv('Euclidean/' + input_csv_file + '.csv', header=None)
    euclidean_mat = df_euclidean.values
    return euclidean_mat


def get_predicted_distance_matrix_DNN_K(input_csv_file):
    df_dnn = pandas.read_csv('Outputs_K/' + input_csv_file + '.csv', header=None)
    dnn_mat = df_dnn.values
    return dnn_mat


def get_region_matrix(input_csv_file):
    df_region = pandas.read_csv('Region_grid/' + input_csv_file + '.csv', header=None)
    region = df_region.values[0]

    region_dict = dict()
    for i in range(len(region)):
        region_dict.setdefault(region[i], []).append(i)

    return region, region_dict


def get_region_matrix_K(input_csv_file):
    df_region = pandas.read_csv('Region_K/' + input_csv_file + '.csv', header=None)
    region = df_region.values[0]

    region_dict = dict()
    for i in range(len(region)):
        region_dict.setdefault(region[i], []).append(i)

    return region, region_dict


coordinates = get_points(map)
actual_distance_matrix = get_actual_distance_matrix(map)
region_array, region_dict = get_region_matrix(map)

region_array_k, region_dict_k = get_region_matrix_K(map)
pprint(region_dict_k)

landmark_ids, predicted_distance_matrix_landmark = get_predicted_distance_matrix_landmark(map, no_landmarks_default)
SE_matrix_landmark = (actual_distance_matrix - predicted_distance_matrix_landmark) ** 2
landmark_coordinates = coordinates[landmark_ids][0]

print("MSE Landmark: {}".format(np.mean(SE_matrix_landmark)))

predicted_distance_matrix_euclidean = get_predicted_distance_matrix_euclidean(map)
SE_matrix_euclidean = (actual_distance_matrix - predicted_distance_matrix_euclidean) ** 2
print("MSE Euclidean: {}".format(np.mean(SE_matrix_euclidean)))

predicted_distance_matrix_EL_max = np.maximum(predicted_distance_matrix_euclidean, predicted_distance_matrix_landmark)
SE_matrix_EL_max = (actual_distance_matrix - predicted_distance_matrix_EL_max) ** 2
print("MSE EL Max: {}".format(np.mean(SE_matrix_EL_max)))

predicted_distance_matrix_dnn_K = get_predicted_distance_matrix_DNN_K(map)
SE_matrix_dnn_K = (actual_distance_matrix - predicted_distance_matrix_dnn_K) ** 2
print("MSE DNN K: {}".format(np.mean(SE_matrix_dnn_K)))

#######################

E_matrix_dnn_K_overestimation = -(predicted_distance_matrix_dnn_K - actual_distance_matrix) * \
                                ((predicted_distance_matrix_dnn_K - actual_distance_matrix) < 0)

overest_nodes = np.zeros(predicted_distance_matrix_EL_max.shape[0])

region_wise_overestimation = np.zeros((int(np.max(region_array_k)+1), int(np.max(region_array_k)+1)))
region_wise_overestimation_c = np.zeros((int(np.max(region_array_k)+1), int(np.max(region_array_k)+1)))

for i in range(E_matrix_dnn_K_overestimation.shape[0]):
    for j in range(E_matrix_dnn_K_overestimation.shape[1]):
        region_wise_overestimation[int(region_array_k[i])][int(region_array_k[j])] += E_matrix_dnn_K_overestimation[i][j]
        region_wise_overestimation_c[int(region_array_k[i])][int(region_array_k[j])] += 1

region_wise_overestimation = np.divide(region_wise_overestimation,region_wise_overestimation_c)

print(region_wise_overestimation)

# print(np.mean(SE_matrix_dnn_K))

# coordinates_overest = np.argwhere(SE_matrix_dnn_K > np.mean(SE_matrix_dnn_K))
# np.savetxt('./overest', E_matrix_dnn_K_overestimation, delimiter=",", fmt='%.3f')

plt.imshow(region_wise_overestimation, cmap='hot', interpolation='nearest')
plt.show()


#######################

def get_nodewise_sq_error(SE_matrix):
    num_nodes = len(SE_matrix)
    sum_error = np.sum(SE_matrix, axis=1)
    sum_error /= num_nodes
    return sum_error


def get_nodewise_overestimation_error_nodes(E_matrix):
    E_matrix = (E_matrix > 1.566)
    sum_error = np.sum(E_matrix, axis=1)
    return sum_error


def draw_heatmap(point_coordinates, error_mat, figure, axis, caption):
    im = axis.scatter(point_coordinates[:, 0], point_coordinates[:, 1], c=error_mat, cmap='viridis')
    figure.colorbar(im, ax=axis)
    axis.title.set_text(caption)

    return


def get_dist_to_max_SE(SE_matrix):
    sum_error = get_nodewise_sq_error(SE_matrix)
    max_err_node_id = np.argmax(sum_error)

    return max_err_node_id, SE_matrix[max_err_node_id]


# sum_error_euclidean = get_nodewise_sq_error(SE_matrix_euclidean)
# plt.plot(np.arange(sum_error_euclidean.shape[0]), sorted(sum_error_euclidean), color='red', label='euclidean')

sum_error_landmark = get_nodewise_sq_error(SE_matrix_landmark)
plt.plot(np.arange(sum_error_landmark.shape[0]), sorted(sum_error_landmark), color='blue', label='landmark')

sum_error_EL_max = get_nodewise_sq_error(SE_matrix_EL_max)
plt.plot(np.arange(sum_error_EL_max.shape[0]), sorted(sum_error_EL_max), color='black', label='EL max')

# sum_error_overest_num = get_nodewise_overestimation_error_nodes(predicted_distance_matrix_dnn_K - actual_distance_matrix)
# plt.plot(np.arange(sum_error_overest_num.shape[0]), sorted(sum_error_overest_num), color='orange', label='overestimation')

sum_error_dnn_K = get_nodewise_sq_error(SE_matrix_dnn_K)
plt.plot(np.arange(sum_error_dnn_K.shape[0]), sorted(sum_error_dnn_K), color='green', label='dnn_K')

# sum_error_dnn_grid = get_nodewise_sq_error(SE_matrix_dnn_grid)
# plt.plot(np.arange(sum_error_dnn_K.shape[0]), sorted(sum_error_dnn_grid), color='purple', label='dnn_grid')

plt.legend()
plt.show()

# sum_error_euclidean = get_nodewise_sq_error(SE_matrix_euclidean)
# plt.plot(np.arange(sum_error_euclidean.shape[0]), sum_error_euclidean, color='red', label='euclidean')

sum_error_landmark = get_nodewise_sq_error(SE_matrix_landmark)
plt.plot(np.arange(sum_error_landmark.shape[0]), sum_error_landmark, color='blue', label='landmark')

sum_error_EL_max = get_nodewise_sq_error(SE_matrix_EL_max)
plt.plot(np.arange(sum_error_EL_max.shape[0]), sum_error_EL_max, color='black', label='EL max')

# sum_error_overest_num = get_nodewise_overestimation_error_nodes(predicted_distance_matrix_dnn_K - actual_distance_matrix)
# plt.plot(np.arange(sum_error_overest_num.shape[0]), sum_error_overest_num, color='orange', label='overestimation')

sum_error_dnn_K = get_nodewise_sq_error(SE_matrix_dnn_K)
plt.plot(np.arange(sum_error_dnn_K.shape[0]), sum_error_dnn_K, color='green', label='dnn_K')

# sum_error_dnn_grid = get_nodewise_sq_error(SE_matrix_dnn_grid)
# plt.plot(np.arange(sum_error_dnn_K.shape[0]), sum_error_dnn_grid, color='purple', label='dnn_grid')

plt.legend()
plt.show()

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 4))

r1, r2 = axes
c1, c2 = r1
c3, c4 = r2

img = plt.imread('Images/' + map + '.png')
c1.imshow(img)
c1.title.set_text('Original Image')

draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_euclidean), fig, c2,
             'euclidean: ' + str(np.mean(SE_matrix_euclidean)))
draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_landmark), fig, c3,
             'landmark: ' + str(np.mean(SE_matrix_landmark)))
# c3.scatter(landmark_coordinates[:, 0], landmark_coordinates[:, 1], c='red', cmap='viridis')


draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_dnn_K), fig, c4, 'dnn_K: ' + str(np.mean(SE_matrix_dnn_K)))
plt.show()
#
# fig_2, axes_2 = plt.subplots(ncols=2, nrows=2, figsize=(8, 4))
#
# r1_2, r2_2 = axes_2
# c1_2, c2_2 = r1_2
# c3_2, c4_2 = r2_2
#
# draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_euclidean), fig_2, c1_2, 'euclidean: ' + str(np.mean(SE_matrix_euclidean)))
#
# max_err_node_id_euclidean, SE_max_euclidean =  get_dist_to_max_SE(SE_matrix_euclidean)
# draw_heatmap(coordinates, SE_max_euclidean, fig_2, c2_2, 'euclidean_max: ' + str(np.mean(SE_max_euclidean)))
# c2_2.scatter(coordinates[max_err_node_id_euclidean][0],coordinates[max_err_node_id_euclidean][1], c='red', cmap='viridis')
#
# draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_landmark), fig_2, c3_2, 'landmark: ' + str(np.mean(SE_matrix_landmark)))
# c3_2.scatter(landmark_coordinates[:, 0], landmark_coordinates[:, 1], c='red', cmap='viridis')
#
# max_err_node_id_landmark, SE_max_landmark = get_dist_to_max_SE(SE_matrix_landmark)
# draw_heatmap(coordinates, SE_max_landmark, fig_2, c4_2, 'landmark_max: ' + str(np.mean(SE_max_landmark)))
# c4_2.scatter(coordinates[max_err_node_id_landmark][0],coordinates[max_err_node_id_landmark][1], c='red', cmap='viridis')
#
# plt.show()
#
# fig_3, axes_3 = plt.subplots(ncols=2, nrows=2, figsize=(8, 4))
#
# r1_3, r2_3 = axes_3
# c1_3, c2_3 = r1_3
# c3_3, c4_3 = r2_3
#
# draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_dnn_K), fig_3, c1_3, 'dnn_K: ' + str(np.mean(SE_matrix_dnn_K)))
#
# max_err_node_id_dnn_K, SE_max_dnn_K = get_dist_to_max_SE(SE_matrix_dnn_K)
# draw_heatmap(coordinates, SE_max_dnn_K, fig_3, c2_3, 'dnn_SE_max: ' + str(np.mean(SE_max_dnn_K)))
# c2_3.scatter(coordinates[max_err_node_id_dnn_K][0],coordinates[max_err_node_id_dnn_K][1], c='red', cmap='viridis')
#
#
# draw_heatmap(coordinates, get_nodewise_sq_error(E_matrix_dnn_K_overestimation), fig_3, c3_3, 'dnn_ov: '
#              + str(np.mean(E_matrix_dnn_K_overestimation)))
#
# max_err_node_id_dnn_K_oe, SE_max_dnn_K_oe = get_dist_to_max_SE(E_matrix_dnn_K_overestimation)
# draw_heatmap(coordinates, SE_max_dnn_K_oe, fig_3, c4_3, 'dnn_SE_max_oe: ' + str(np.mean(SE_max_dnn_K_oe)))
# c4_3.scatter(coordinates[max_err_node_id_dnn_K_oe][0],coordinates[max_err_node_id_dnn_K_oe][1], c='red', cmap='viridis')
#
# plt.show()
#
# fig_4, axes_4 = plt.subplots(ncols=2, nrows=2, figsize=(8, 4))
#
# r1_4, r2_4 = axes_4
# c1_4, c2_4 = r1_4
# c3_4, c4_4 = r2_4
#
# draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_euclidean), fig_4, c1_4, 'euclidean: ' + str(np.mean(SE_matrix_euclidean)))
#
# landmark_ids_4, predicted_distance_matrix_landmark_4 = get_predicted_distance_matrix_landmark(map, 4)
# SE_matrix_landmark_4 = (actual_distance_matrix - predicted_distance_matrix_landmark_4) ** 2
# landmark_coordinates_4 = coordinates[landmark_ids_4][0]
#
# draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_landmark_4), fig_4, c2_4, 'landmark_4: ' + str(np.mean(SE_matrix_landmark_4)))
# c2_4.scatter(landmark_coordinates_4[:, 0], landmark_coordinates_4[:, 1], c='red', cmap='viridis')
#
# landmark_ids_8, predicted_distance_matrix_landmark_8 = get_predicted_distance_matrix_landmark(map, 8)
# SE_matrix_landmark_8 = (actual_distance_matrix - predicted_distance_matrix_landmark_8) ** 2
# landmark_coordinates_8 = coordinates[landmark_ids_8][0]
#
# draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_landmark_8), fig_4, c3_4, 'landmark_8: ' + str(np.mean(SE_matrix_landmark_8)))
# c3_4.scatter(landmark_coordinates_8[:, 0], landmark_coordinates_8[:, 1], c='red', cmap='viridis')
#
# landmark_ids_12, predicted_distance_matrix_landmark_12 = get_predicted_distance_matrix_landmark(map, 12)
# SE_matrix_landmark_12 = (actual_distance_matrix - predicted_distance_matrix_landmark_12) ** 2
# landmark_coordinates_12 = coordinates[landmark_ids_12][0]
#
# draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_landmark_12), fig_4, c4_4, 'landmark_12: ' + str(np.mean(SE_matrix_landmark_12)))
# c4_4.scatter(landmark_coordinates_12[:, 0], landmark_coordinates_12[:, 1], c='red', cmap='viridis')
#
# plt.show()
#
# fig_5, axes_5 = plt.subplots(ncols=2, nrows=2, figsize=(8, 4))
#
# r1_5, r2_5 = axes_5
# c1_5, c2_5 = r1_5
# c3_5, c4_5 = r2_5
#
# draw_heatmap(coordinates, SE_max_euclidean, fig_5, c1_5, 'euclidean_max: ' + str(np.mean(SE_max_euclidean)))
# c1_5.scatter(coordinates[max_err_node_id_euclidean][0],coordinates[max_err_node_id_euclidean][1], c='red', cmap='viridis')
#
# max_err_node_id_landmark_4, SE_max_landmark_4 = get_dist_to_max_SE(SE_matrix_landmark_4)
# draw_heatmap(coordinates, SE_max_landmark_4, fig_5, c2_5, 'landmark_max_4: ' + str(np.mean(SE_max_landmark_4)))
# c2_5.scatter(coordinates[max_err_node_id_landmark_4][0],coordinates[max_err_node_id_landmark_4][1], c='red', cmap='viridis')
#
# max_err_node_id_landmark_8, SE_max_landmark_8 = get_dist_to_max_SE(SE_matrix_landmark_8)
# draw_heatmap(coordinates, SE_max_landmark_8, fig_5, c3_5, 'landmark_max_8: ' + str(np.mean(SE_max_landmark_8)))
# c3_5.scatter(coordinates[max_err_node_id_landmark_8][0],coordinates[max_err_node_id_landmark_8][1], c='red', cmap='viridis')
#
# max_err_node_id_landmark_12, SE_max_landmark_12 = get_dist_to_max_SE(SE_matrix_landmark_12)
# draw_heatmap(coordinates, SE_max_landmark_12, fig_5, c4_5, 'landmark_max_12: ' + str(np.mean(SE_max_landmark_12)))
# c4_5.scatter(coordinates[max_err_node_id_landmark_12][0],coordinates[max_err_node_id_landmark_12][1], c='red', cmap='viridis')
#
# plt.show()
