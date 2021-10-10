import glob
import os
import pickle
import pprint
import random

import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_generator_3 import load_dataset, dataset_loader, dataset_container
from map_test import get_actual_distance_matrix, get_distance_matrix_landmark

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map = 'brc300d'
input_csv_file = map

TOTAL_EPOCHS = 70
PASSAWAY_EPOCHS = 30
OBSERVE_EPOCHS = TOTAL_EPOCHS - PASSAWAY_EPOCHS

df_euclidean = pandas.read_csv('Euclidean/' + input_csv_file + '.csv', header=None)
df_region = pandas.read_csv("Region_K" + "/" + input_csv_file + ".csv", header=None)
df_euclidean_np = df_euclidean.values
region = df_region.values[0]

original_full_training_data, num_nodes, num_regions, max_val, min_val = load_dataset(input_csv_file)
actual_full_distance_matrix = get_actual_distance_matrix(map)

original_full_training_data_x, original_full_training_data_y = original_full_training_data[:, :-1], \
                                                               original_full_training_data[:, -1]

full_ds = dataset_loader(original_full_training_data_x, original_full_training_data_x)


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


def model_train(train_ds, index_no, no_epochs=100, pretrained_model=False):
    print("\n")
    print("Index: " + str(index_no))
    print("DS Length: {}".format(train_ds.__len__()))

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    input_vec_shape = train_ds.x.shape[1]

    if not pretrained_model:
        model = leaf_dnn(input_vec_shape=input_vec_shape,
                         total_num_nodes=num_nodes,
                         total_regions=num_regions,
                         embedding_dim_nodes=128,
                         embedding_dim_regions=32)
    else:
        model = torch.load(os.path.join('temporary_models', str('model_' + str(index_no))))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    min_train_loss = 1000000
    min_train_loss_ov = -1
    min_train_loss_und = -1
    train_batch_num = 0

    for epoch in range(no_epochs):
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

            # how many queries are overestimating / underestimating
            overestimation += torch.sum(torch.ones(y_t.shape).to(device) * ((y_t + 0.00001) < y_pred)).item()
            underestimation += torch.sum(torch.ones(y_t.shape).to(device) * ((y_t - 0.00001) > y_pred)).item()

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= (train_batch_num + 1)

        if epoch % 1 == 0:
            print("epoch {}; T loss={:.9f}; overestimation: {}; underestimation: {}".
                  format(epoch, train_loss, overestimation, underestimation))

        if epoch == 0 or min_train_loss > train_loss:
            min_train_loss = train_loss
            min_train_loss_ov = overestimation
            min_train_loss_und = underestimation
            torch.save(model, os.path.join('temporary_models', str('model_' + str(index_no))))

        if overestimation < 0.1:
            min_train_loss = 0
            torch.save(model, os.path.join('temporary_models', str('model_' + str(index_no))))
            return min_train_loss, overestimation, underestimation

    return min_train_loss, min_train_loss_ov, min_train_loss_und


def infer_distance_from_model(index_no):
    model = torch.load(os.path.join('temporary_models', str('model_' + str(index_no))))
    adj_mat_model = np.zeros((num_nodes, num_nodes))

    batch_rows = 16
    test_loader = DataLoader(full_ds, batch_size=num_nodes * batch_rows, shuffle=False)

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

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adj_mat_model[j, i] = adj_mat_model[i, j]

    return adj_mat_model


def infer_distance_from_ensemble(index_node_dict):
    network_index = np.ones(num_nodes) * -1

    networkless_nodes = False
    for k, v in index_node_dict.items():
        if k == -1:
            networkless_nodes = True
        for i in range(len(v)):
            network_index[int(v[i])] = k

    adj_mat_model = np.zeros((num_nodes, num_nodes))
    num_networks = len(index_node_dict.keys()) - 1

    if not networkless_nodes:
        num_networks += 1

    final_adj_mat_model = np.zeros((num_nodes,
                                    num_nodes))

    from_euclidean = np.zeros((num_nodes, num_nodes))

    for i in range(len(adj_mat_model)):
        for j in range(len(adj_mat_model)):
            if region[i] == region[j] or network_index[i] != network_index[j] \
                    or network_index[i] == -1 or network_index[j] == -1:
                from_euclidean[i, j] = 1

    final_adj_mat_model += np.multiply(from_euclidean, df_euclidean_np)

    for k in range(num_networks):
        from_model_k = np.zeros((num_nodes, num_nodes))
        for i in range(len(adj_mat_model)):
            for j in range(len(adj_mat_model)):
                if from_euclidean[i, j] == 0 and network_index[i] == network_index[j] and network_index[i] == k:
                    from_model_k[i, j] = 1

        adj_mat_model = infer_distance_from_model(k)
        final_adj_mat_model += np.multiply(from_model_k, adj_mat_model)

    np.savetxt('final.npy', final_adj_mat_model, delimiter=",", fmt='%.3f')

    return final_adj_mat_model


def find_nodewise_error(actual_matrix, predicted_matrix):
    diff_matrix = np.round(actual_matrix, 3) - np.round(predicted_matrix, 3)
    diff_matrix_ov = np.ones(actual_matrix.shape) * (diff_matrix < 0)
    nodewise_error = np.sum(np.abs(np.multiply(diff_matrix_ov, diff_matrix)),
                            axis=1)

    return nodewise_error


def get_error(actual_matrix, predicted_matrix_):
    diff_matrix = np.round(actual_matrix, 3) - np.round(predicted_matrix_, 3)
    return np.sum(np.abs(diff_matrix))


def get_error_ov(actual_matrix, predicted_matrix_):
    diff_matrix = np.round(actual_matrix, 3) - np.round(predicted_matrix_, 3)
    diff_matrix_ov = np.ones(actual_matrix.shape) * (diff_matrix < 0)
    if np.count_nonzero(diff_matrix_ov) > 0:
        return np.sum(np.abs(np.multiply(diff_matrix_ov, diff_matrix)))
    return 0


def get_error_und(actual_matrix, predicted_matrix_):
    diff_matrix = np.round(actual_matrix, 3) - np.round(predicted_matrix_, 3)
    diff_matrix_und = np.ones(actual_matrix.shape) * (diff_matrix > 0)
    if np.count_nonzero(diff_matrix_und) > 0:
        return np.sum(np.abs(np.multiply(diff_matrix_und, diff_matrix)))
    return 0


def get_reduced_distance_matrix(distance_matrix_, nodes_list_):
    nodes_list_ = list(nodes_list_)
    distance_matrix_col = distance_matrix_[:, nodes_list_]
    distance_matrix_row = distance_matrix_col[nodes_list_, :]

    return distance_matrix_row


def get_node_partition(predicted_matrix, nodes_list_):
    reduced_actual_matrix = get_reduced_distance_matrix(actual_full_distance_matrix, nodes_list_)
    reduced_predicted_matrix = get_reduced_distance_matrix(predicted_matrix, nodes_list_)
    nodewise_error = find_nodewise_error(reduced_actual_matrix, reduced_predicted_matrix)

    nodewise_error_indexed = list(zip(nodes_list_, nodewise_error))
    nodewise_error_indexed_sorted = sorted(nodewise_error_indexed, key=lambda item: item[1])

    threshold_value = 0.8
    num_nodes_included = int(len(nodes_list_) * threshold_value)

    included_nodes_and_errors = nodewise_error_indexed_sorted[:num_nodes_included]
    excluded_nodes_and_errors = nodewise_error_indexed_sorted[num_nodes_included:]

    included_nodes = []
    excluded_nodes = []

    for i in range(len(included_nodes_and_errors)):
        included_nodes.append(included_nodes_and_errors[i][0])

    for i in range(len(excluded_nodes_and_errors)):
        excluded_nodes.append(excluded_nodes_and_errors[i][0])

    included_nodes = np.sort(np.array(included_nodes))
    excluded_nodes = np.sort(np.array(excluded_nodes))

    return included_nodes, excluded_nodes


def train_single_index_model(full_ds_container, original_training_ds, original_nodes_list, index_no):
    included_nodes_list = list(original_nodes_list)
    excluded_nodes_list = []

    train_loss, overestimation, underestimation = model_train(train_ds=original_training_ds,
                                                              index_no=index_no,
                                                              no_epochs=TOTAL_EPOCHS,
                                                              pretrained_model=False)

    counter = 1
    while train_loss != 0:
        predicted_matrix = infer_distance_from_model(index_no)
        new_included_nodes, new_excluded_nodes = get_node_partition(predicted_matrix,
                                                                    included_nodes_list)
        included_nodes_list = list(new_included_nodes)
        excluded_nodes_list = excluded_nodes_list + (list(new_excluded_nodes))

        new_training_ds_x, new_training_ds_y = full_ds_container.new_training_ds(included_nodes_list)

        new_training_ds = dataset_loader(new_training_ds_x, new_training_ds_y)

        train_loss, overestimation, underestimation = model_train(train_ds=new_training_ds,
                                                                  index_no=index_no,
                                                                  no_epochs=PASSAWAY_EPOCHS + counter * 20,
                                                                  pretrained_model=True)

        counter += 1

    excluded_nodes_list = sorted(excluded_nodes_list)
    return included_nodes_list, excluded_nodes_list


def controller(num_dnn):
    files = glob.glob('temporary_models/*')
    for f_ in files:
        os.remove(f_)

    full_ds_container = dataset_container(original_full_training_data, num_nodes)
    nodes_list = np.arange(num_nodes)

    included_nodes_list = nodes_list
    training_ds_x, training_ds_y = full_ds_container.new_training_ds(included_nodes_list)
    training_ds = dataset_loader(training_ds_x, training_ds_y)

    node_nn_dict_ = dict()

    for i in range(num_dnn):
        new_included_nodes_list, new_excluded_nodes_list = train_single_index_model(full_ds_container,
                                                                                    index_no=i,
                                                                                    original_training_ds=training_ds,
                                                                                    original_nodes_list=
                                                                                    included_nodes_list)
        node_nn_dict_[i] = new_included_nodes_list

        included_nodes_list = list(new_excluded_nodes_list)
        print(len(included_nodes_list))
        if len(included_nodes_list) == 0:
            break
        training_ds_x, training_ds_y = full_ds_container.new_training_ds(included_nodes_list)

        if len(training_ds_x) == 0:
            break

        training_ds = dataset_loader(training_ds_x, training_ds_y)

        if i == num_dnn - 1:
            node_nn_dict_[-1] = new_excluded_nodes_list

    return node_nn_dict_


node_nn_dict = controller(5)
pp = pprint.PrettyPrinter(indent=4, width=250)
pp.pprint(node_nn_dict)

with open(os.path.join('temporary_models', 'dict.pkl'), 'wb') as f:
    pickle.dump(node_nn_dict, f)

print("Euclidean Error : {}, Overestimation Error : {}, Underestimation Error : {}".format(
    get_error(actual_full_distance_matrix, df_euclidean_np),
    get_error_ov(actual_full_distance_matrix, df_euclidean_np),
    get_error_und(actual_full_distance_matrix, df_euclidean_np)
))

np.savetxt('euclidean.npy', df_euclidean_np, delimiter=",", fmt='%.3f')

landmark_np = get_distance_matrix_landmark(map, 12)
print("Landmark Error : {}, Overestimation Error : {}, Underestimation Error : {}".format(
    get_error(actual_full_distance_matrix, landmark_np),
    get_error_ov(actual_full_distance_matrix, landmark_np),
    get_error_und(actual_full_distance_matrix, landmark_np)
))

with open(os.path.join('temporary_models', 'dict.pkl'), 'rb') as f:
    node_nn_dict = pickle.load(f)

for k, v in node_nn_dict.items():
    print("{} : {} Nodes".format(k, len(v)))

predicted_matrix = infer_distance_from_ensemble(node_nn_dict)
print("NN Error : {}, Overestimation Error : {}, Underestimation Error : {}".format(
    get_error(actual_full_distance_matrix, predicted_matrix),
    get_error_ov(actual_full_distance_matrix, predicted_matrix),
    get_error_und(actual_full_distance_matrix, predicted_matrix)
))

# np.savetxt('euclidean.npy', (predicted_matrix - df_euclidean_np), delimiter=",", fmt='%.3f')
# np.savetxt('error.npy', (predicted_matrix - actual_full_distance_matrix), delimiter=",", fmt='%.3f')


# p_m = infer_distance_from_model(3)
# p_m_r = get_reduced_distance_matrix(p_m, [0, 3, 9, 18])
# print(p_m_r)

# dataset_container(original_full_training_data, num_nodes, min_val, max_val).new_training_ds(node_nn_dict[1],
#                                                                                             min_val, max_val)
