import glob
import os
import random

import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_generator_2 import load_dataset, dataset_loader
from map_test import get_actual_distance_matrix, get_predicted_distance_matrix_DNN_K, get_distance_matrix_euclidean, \
    get_distance_matrix_landmark

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map = 'brc300d'
input_csv_file = map

TOTAL_EPOCHS = 7
PASSAWAY_EPOCHS = 3
OBSERVE_EPOCHS = TOTAL_EPOCHS - PASSAWAY_EPOCHS

df_euclidean = pandas.read_csv('Euclidean/' + input_csv_file + '.csv', header=None)
df_region = pandas.read_csv("Region_K" + "/" + input_csv_file + ".csv", header=None)
df_euclidean_np = df_euclidean.values
region = df_region.values[0]

original_full_training_data, num_nodes, num_regions, max_val, min_val = load_dataset(input_csv_file)
actual_distances_full = get_actual_distance_matrix(map)
full_ds = dataset_loader(original_full_training_data, train_mode=True)


def infer_distances_2(model_file_name, save_path=None):
    model = torch.load(os.path.join('temporary_models', model_file_name))
    adj_mat_model = np.zeros((num_nodes, num_nodes))
    from_euclidean = np.zeros((num_nodes, num_nodes))
    from_model = np.zeros((num_nodes, num_nodes))

    for i in range(len(adj_mat_model)):
        for j in range(len(adj_mat_model)):
            if region[i] == region[j]:
                from_euclidean[i, j] = 1
            else:
                from_model[i, j] = 1

    batch_rows = 16
    ds = dataset_loader(original_full_training_data, train_mode=False)
    test_loader = DataLoader(ds, batch_size=num_nodes * batch_rows, shuffle=False)

    for batch_idx, (x, y_t) in enumerate(test_loader):
        x_coord = x[:, :2]
        x_region = x[:, 2:4]
        x_inp = x[:, 4:]

        x_coord = np.array(x_coord).astype(np.int64)
        x_coord = torch.LongTensor(x_coord).to(device)
        x_region = np.array(x_region).astype(np.int64)
        x_region = torch.LongTensor(x_region).to(device)
        x_inp = np.array(x_inp).astype(np.float32)
        x_inp = torch.FloatTensor(x_inp).to(device)

        y_pred = model(x_coord, x_region, x_inp)

        adj_mat = torch.reshape(y_pred, shape=(-1, num_nodes))
        adj_mat = adj_mat.cpu().detach().numpy()
        adj_mat = adj_mat * (max_val - min_val) + min_val
        adj_mat_model[batch_rows * batch_idx: batch_rows * (batch_idx + 1), :] = adj_mat

    adj_mat_total = np.multiply(from_euclidean, df_euclidean_np) + np.multiply(from_model, adj_mat_model)

    if save_path is not None:
        np.savetxt(save_path, adj_mat_total, delimiter=",", fmt='%.3f')

    return adj_mat_total


class leaf_dnn(nn.Module):

    def __init__(self, input_vec_shape, total_num_nodes, total_regions, embedding_dim_nodes=128,
                 embedding_dim_regions=32):
        super(leaf_dnn, self).__init__()
        self.embedding_dim = embedding_dim_nodes
        self.node_embeddings = nn.Embedding(total_num_nodes, embedding_dim_nodes)
        self.region_embeddings = nn.Embedding(total_regions, embedding_dim_regions)
        self.input_vec_shape = input_vec_shape

        self.fc1 = nn.Linear(self.input_vec_shape - 4 + 2 * embedding_dim_nodes + 2 * embedding_dim_regions,
                             embedding_dim_nodes)
        self.fc2 = nn.Linear(embedding_dim_nodes, embedding_dim_nodes // 2)
        self.fc3 = nn.Linear(embedding_dim_nodes // 2, embedding_dim_nodes // 4)
        self.fc4 = nn.Linear(embedding_dim_nodes // 4, embedding_dim_nodes // 8)
        self.fc5 = nn.Linear(embedding_dim_nodes // 8, 1)

    def forward(self, x_node_id, x_region_id, x_inp):
        embed_node = self.node_embeddings(x_node_id)
        embed_region = self.region_embeddings(x_region_id)
        embed_node_r = torch.reshape(embed_node, (x_node_id.shape[0], -1))
        embed_region_r = torch.reshape(embed_region, (x_region_id.shape[0], -1))
        embed_r_c = torch.cat([embed_node_r, embed_region_r, x_inp], dim=-1)

        out_1 = torch.relu(self.fc1(embed_r_c))
        out_2 = torch.relu(self.fc2(out_1))
        out_3 = torch.relu(self.fc3(out_2))
        out_4 = torch.relu(self.fc4(out_3))
        out_5 = self.fc5(out_4)

        return out_5


def ensemble_learning_model():
    def train_model(train_ds, model_file_name, epochs, pretrained_model=None):

        print("\n")
        print("Model_file_name: " + str(model_file_name))
        print("DS Length: {}".format(train_ds.__len__()))

        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
        input_vec_shape = train_ds.x.shape[1]

        if pretrained_model is None:
            model = leaf_dnn(input_vec_shape=input_vec_shape,
                             total_num_nodes=num_nodes,
                             total_regions=num_regions,
                             embedding_dim_nodes=128,
                             embedding_dim_regions=32)
        else:
            model = torch.load(os.path.join('temporary_models', pretrained_model))

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        min_train_loss = 1000000
        train_batch_num = 0

        overestimation_matrix = np.zeros((num_nodes, num_nodes))
        underestimation_matrix = np.zeros((num_nodes, num_nodes))

        for epoch in range(epochs):
            train_loss = 0.0
            underestimation = 0
            overestimation = 0

            model.train()

            for batch_idx, (x, y_t) in enumerate(train_loader):
                x_coord = x[:, :2]
                x_region = x[:, 2:4]
                x_inp = x[:, 4:]

                x_coord = np.array(x_coord).astype(np.int64)
                x_coord = torch.LongTensor(x_coord).to(device)

                x_region = np.array(x_region).astype(np.int64)
                x_region = torch.LongTensor(x_region).to(device)

                x_inp = np.array(x_inp).astype(np.float32)
                x_inp = torch.FloatTensor(x_inp).to(device)

                y_t = torch.FloatTensor(y_t).to(device)

                train_batch_num = batch_idx
                optimizer.zero_grad()
                y_pred = model(x_coord, x_region, x_inp)
                y_pred = torch.squeeze(y_pred)

                loss = torch.nn.MSELoss().to(device)(y_t, y_pred)

                overestimation += torch.sum(torch.ones(y_t.shape).to(device) * ((y_t + 0.001) < y_pred)).item()
                underestimation += torch.sum(torch.ones(y_t.shape).to(device) * ((y_t - 0.001) > y_pred)).item()

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            train_loss /= (train_batch_num + 1)

            if epoch % 1 == 0:
                print("epoch {}; T loss={:.9f}; overestimation: {}; underestimation: {}".
                      format(epoch, train_loss, overestimation, underestimation))

            if epoch == 0 or min_train_loss > train_loss:
                min_train_loss = train_loss
                torch.save(model, os.path.join('temporary_models', model_file_name))

            if epoch >= PASSAWAY_EPOCHS:
                predicted_adj_mat_epoch = infer_distances_2(model_file_name)
                error_matrix_epoch = get_difference_signed(actual_distances_full, predicted_adj_mat_epoch)
                underestimation_matrix += np.array((error_matrix_epoch < -0.00001)).astype(int)
                overestimation_matrix += np.array((error_matrix_epoch > 0.00001)).astype(int)

        print("Done")
        print("\n\n")
        # print("Overestimation: ")
        # print(overestimation_matrix[:20, :20])
        # print("Underestimation:")
        # print(underestimation_matrix)
        # print("\n\n")
        #
        # np.savetxt("underestimation.npy", underestimation_matrix, delimiter=",", fmt="%.0f")
        # np.savetxt("overestimation.npy", overestimation_matrix, delimiter=",", fmt="%.0f")

        return overestimation_matrix, underestimation_matrix

    def get_difference_signed(original_matrix, predicted_matrix):
        diff_matrix = np.round(original_matrix, 3) - np.round(predicted_matrix, 3)
        return diff_matrix

    files = glob.glob('temporary_models/*')
    for f in files:
        os.remove(f)

    files = glob.glob('temporary_datasets/*')
    for f in files:
        os.remove(f)

    files = glob.glob('estimation_counts/*')
    for f in files:
        os.remove(f)

    current_training_data = original_full_training_data
    model_queue = []

    for i in range(4):

        if i == 0:
            ov_mat, und_mat = train_model(full_ds,
                                          model_file_name=str('temp_model_' + str('p_0')),
                                          epochs=TOTAL_EPOCHS,
                                          pretrained_model=None)

            np.savetxt(os.path.join('estimation_counts', 'ov_p_0.npy'), ov_mat, delimiter=",", fmt="%.0f")
            np.savetxt(os.path.join('estimation_counts', 'und_p_0.npy'), und_mat, delimiter=",", fmt="%.0f")

            non_mat_mask = (ov_mat < OBSERVE_EPOCHS) & (und_mat < OBSERVE_EPOCHS)

            full_ds.generate_new_training_ds(non_mat_mask, str(map + '_no_' + str(i) + '.csv'))
            model_queue.append(['no_0', 'p_0'])

        else:
            popped_val = model_queue.pop()
            model_file_suffix = popped_val[0]
            parent_suffix = popped_val[1]

            ds = dataset_loader(training_ds=None, train_mode=True, read_from_csv=str(map + '_' + str(model_file_suffix)
                                                                                     + '.csv'))

            ov_mat, und_mat = train_model(ds,
                                          model_file_name=str('temp_model_' + str(model_file_suffix)),
                                          epochs=TOTAL_EPOCHS,
                                          pretrained_model='temp_model_' + str(parent_suffix))

            np.savetxt(os.path.join('estimation_counts', 'ov_' + model_file_suffix + '.npy'),
                       ov_mat, delimiter=",", fmt="%.0f")
            np.savetxt(os.path.join('estimation_counts', 'und_' + model_file_suffix + '.npy'),
                       und_mat, delimiter=",", fmt="%.0f")

            non_mat_mask = (ov_mat < OBSERVE_EPOCHS) & (und_mat < OBSERVE_EPOCHS)
            full_ds.generate_new_training_ds(non_mat_mask, str(map + '_no_' + str(i) + '.csv'))

            model_queue.append([str('no_' + str(i)), model_file_suffix])

    return


ensemble_learning_model()

actual_distance = get_actual_distance_matrix(map)

euclidean_distance = get_distance_matrix_euclidean(map)
adjacency_matrix = np.abs(np.round(actual_distance, 3) - np.round(euclidean_distance, 3))

print("Euclidean: ")
print(np.mean(adjacency_matrix))
print(np.std(adjacency_matrix))
print("\n\n")

landmark_distance = get_distance_matrix_landmark(map, 12)
adjacency_matrix = np.abs(np.round(actual_distance, 3) - np.round(landmark_distance, 3))

print("Landmark: ")
print(np.mean(adjacency_matrix))
print(np.std(adjacency_matrix))
print("\n\n")

adjacency_matrix = np.abs(np.round(actual_distance, 3) -
                          np.round(np.maximum(euclidean_distance, landmark_distance), 3))

print("Max(Euclidean,Landmark):")
print(np.mean(adjacency_matrix))
print(np.std(adjacency_matrix))
print("\n\n")

predicted_distance_matrix_dnn_K = get_predicted_distance_matrix_DNN_K(map)
adjacency_matrix = np.abs(np.round(actual_distance, 3) - np.round(predicted_distance_matrix_dnn_K, 3))

print("NN: ")
print(np.mean(adjacency_matrix))
print(np.std(adjacency_matrix))
print("\n\n")
