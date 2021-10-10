import glob
import os
import pickle
import random

import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_generator_4 import load_dataset, dataset_loader, dataset_container
from map_test import get_actual_distance_matrix, get_distance_matrix_landmark

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map = 'brc300d'
input_csv_file = map

TOTAL_EPOCHS = 70

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


def update_weights(full_ds_container, old_weight, old_ov_mask, model_index_no):
    predicted_matrix_ = infer_distance_from_model(model_index_no)
    ov_error_matrix, ov_error_mask = get_error_ov(actual_full_distance_matrix, predicted_matrix_)

    new_ov_mask = np.logical_and(old_ov_mask, ov_error_mask)
    new_weight_matrix = np.multiply(new_ov_mask, ov_error_matrix)
    new_weight_matrix_train = full_ds_container.get_weights(new_weight_matrix)
    new_weight = old_weight + new_weight_matrix_train
    new_weight = new_weight * (len(new_weight) / np.sum(new_weight))

    return new_weight, new_ov_mask


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
        model = torch.load(os.path.join('temporary_models', str('model_' + str(index_no - 1))))

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

        for batch_idx, (x, y_t, w) in enumerate(train_loader):
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
            w = torch.FloatTensor(w).to(device)

            train_batch_num = batch_idx
            optimizer.zero_grad()
            y_pred = model(x_coord, x_region, x_inp)
            y_pred = torch.squeeze(y_pred)

            loss = torch.nn.MSELoss(reduction='none').to(device)(y_t, y_pred)
            loss = loss * w

            # how many queries are overestimating / underestimating
            overestimation += torch.sum(torch.ones(y_t.shape).to(device) * ((y_t + 0.00001) < y_pred)).item()
            underestimation += torch.sum(torch.ones(y_t.shape).to(device) * ((y_t - 0.00001) > y_pred)).item()

            train_loss += loss.mean().item()

            loss.mean().backward()
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

        if min_train_loss == 0:
            min_train_loss = 0
            torch.save(model, os.path.join('temporary_models', str('model_' + str(index_no))))
            return min_train_loss, overestimation, underestimation

    return min_train_loss, min_train_loss_ov, min_train_loss_und


def infer_distance_from_model(index_no):
    model = torch.load(os.path.join('temporary_models', str('model_' + str(index_no))))
    adj_mat_model = np.zeros((num_nodes, num_nodes))

    batch_rows = 16
    test_loader = DataLoader(full_ds, batch_size=num_nodes * batch_rows, shuffle=False)

    for batch_idx, (x, y_t, _) in enumerate(test_loader):
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


def infer_distance_from_ensemble(num_networks):
    final_adj_mat_model = np.zeros((num_nodes, num_nodes))
    for i in range(num_networks):
        adj_mat_model = infer_distance_from_model(i)
        if i == 0:
            final_adj_mat_model = adj_mat_model
        else:
            final_adj_mat_model = np.minimum(final_adj_mat_model, adj_mat_model)

    from_euclidean = np.zeros((num_nodes, num_nodes))
    from_model = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if region[i] == region[j]:
                from_euclidean[i, j] = 1
            else:
                from_model[i, j] = 1

    return np.multiply(final_adj_mat_model, from_model) + np.multiply(df_euclidean_np, from_euclidean)


def get_error(actual_matrix, predicted_matrix_):
    diff_matrix = np.round(actual_matrix, 3) - np.round(predicted_matrix_, 3)
    diff_matrix = np.square(diff_matrix)
    return diff_matrix


def get_error_ov(actual_matrix, predicted_matrix_):
    diff_matrix = np.round(predicted_matrix_, 3) - np.round(actual_matrix, 3)
    diff_matrix_mask = np.ones(actual_matrix.shape) * (diff_matrix > 0)
    return np.square(np.multiply(diff_matrix, diff_matrix_mask)), diff_matrix_mask


def get_error_und(actual_matrix, predicted_matrix_):
    diff_matrix = np.round(predicted_matrix_, 3) - np.round(actual_matrix, 3)
    diff_matrix_mask = np.ones(actual_matrix.shape) * (diff_matrix < 0)
    return np.square(np.multiply(diff_matrix, diff_matrix_mask)), diff_matrix_mask


def controller(num_networks):
    files = glob.glob('temporary_models/*')
    for f_ in files:
        os.remove(f_)

    full_ds_container = dataset_container(original_full_training_data, num_nodes)
    x, y = full_ds_container.get_dataset()

    weights = np.ones(y.shape).astype(np.float32)
    ov_mask = np.ones((num_nodes, num_nodes))

    for i in range(num_networks):
        print("I: {}, OV: {}".format(i, np.sum(ov_mask)))
        if i >= 1:
            model_train(train_ds=dataset_loader(x, y, weights),
                        index_no=i, no_epochs=TOTAL_EPOCHS, pretrained_model=True)
            weights, ov_mask = update_weights(full_ds_container, weights, ov_mask, i)

        else:
            model_train(train_ds=dataset_loader(x, y, weights),
                        index_no=i, no_epochs=TOTAL_EPOCHS, pretrained_model=False)
            weights, ov_mask = update_weights(full_ds_container, weights, ov_mask, i)

    print("Final: {}".format(np.sum(ov_mask)))
    return


# controller(3)

print("Euclidean Error : {}, Overestimation Error : {}, Underestimation Error : {}".format(
    np.mean(get_error(actual_full_distance_matrix, df_euclidean_np)),
    np.mean(get_error_ov(actual_full_distance_matrix, df_euclidean_np)[0]),
    np.mean(get_error_und(actual_full_distance_matrix, df_euclidean_np)[0])
))

np.savetxt('euclidean.npy', df_euclidean_np, delimiter=",", fmt='%.3f')

landmark_np = get_distance_matrix_landmark(map, 12)
print("Landmark Error : {}, Overestimation Error : {}, Underestimation Error : {}".format(
    np.mean(get_error(actual_full_distance_matrix, landmark_np)),
    np.mean(get_error_ov(actual_full_distance_matrix, landmark_np)[0]),
    np.mean(get_error_und(actual_full_distance_matrix, landmark_np)[0])
))

predicted_matrix = infer_distance_from_ensemble(3)

ov_errors, ov_mask = get_error_ov(actual_full_distance_matrix, predicted_matrix)
max = np.max(ov_errors)
sum = np.sum(ov_mask)
avg = np.mean(ov_errors)

und_errors, und_max = get_error_und(actual_full_distance_matrix, predicted_matrix)
max_u = np.max(und_errors)
sum_u = np.sum(und_max)
avg_u = np.mean(und_errors)

print("NN Error : {}\n Max OV: {}, Num OV: {}, Avg. OV: {}\n Max UND: {}, Num UND: {}, Avg. UND: {}".format(
    np.mean(get_error(actual_full_distance_matrix, predicted_matrix)), max, sum, avg,
    max_u, sum_u, avg_u
))

predicted_matrix = predicted_matrix - max

print("Error after subtracting max OV: {}".
      format(np.mean(get_error(actual_full_distance_matrix, predicted_matrix)[0])))


np.savetxt('final.npy', predicted_matrix, delimiter=",", fmt='%.3f')
