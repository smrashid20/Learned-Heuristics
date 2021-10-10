import numpy as np
import pandas
from torch.utils.data import Dataset

Region_type = 'K'


def norm_minmax(df, max_val=None, min_val=None):
    if max_val is None:
        max_val = np.max(df)
    if min_val is None:
        min_val = np.min(df)
    df = (df - min_val) / (max_val - min_val)
    return df


def load_dataset(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_euclidean = pandas.read_csv('Euclidean/' + input_csv_file + '.csv', header=None)
    df_region = pandas.read_csv("Region_" + Region_type + "/" + input_csv_file + ".csv", header=None)

    df_np = df.values
    df_euclidean_np = df_euclidean.values

    region = df_region.values[0]
    num_regions = int(np.max(region)) + 1

    max_val = np.max(df_np)
    min_val = np.min(df_np)

    df_np = norm_minmax(df_np, min_val=min_val, max_val=max_val)
    df_euclidean_np = norm_minmax(df_euclidean_np, min_val=min_val, max_val=max_val)

    num_nodes = df_np.shape[0]
    all_region = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            all_region.append([int(region[i]), int(region[j])])

    all_region = np.array(all_region)
    a = np.reshape(np.indices(df_np.shape), newshape=(2, -1)).T

    df_np = df_np.reshape(-1, 1)
    df_euclidean_np = df_euclidean_np.reshape(-1, 1)

    training_ds = np.concatenate([a, all_region, df_euclidean_np, df_np], axis=1)

    return training_ds, num_nodes, num_regions, max_val, min_val


class dataset_container:

    def __init__(self, full_training_ds, total_num_nodes):
        self.x, self.y = full_training_ds[:, :-1], full_training_ds[:, -1]
        self.x = np.array(self.x).astype(np.float32)
        self.y = np.array(self.y).astype(np.float32)
        self.total_num_nodes = total_num_nodes

        x_coord = self.x[:, :2]
        x_train = []
        y_train = []

        for i in range(len(x_coord)):
            coo = x_coord[i]
            if coo[0] <= coo[1] and abs(self.x[i][2] - self.x[i][3]) > 0.1:
                x_train.append(self.x[i])
                y_train.append(self.y[i])

        self.x_train_full = np.array(x_train).astype(np.float32)
        self.y_train_full = np.array(y_train).astype(np.float32)

    def get_dataset(self):
        return self.x_train_full, self.y_train_full

    def get_weights(self, full_weight_matrix):
        x_coord = self.x[:, :2]
        weights = []
        for i in range(len(self.x_train_full)):
            s, d = self.x_train_full[i][0], self.x_train_full[i][1]
            weights.append(full_weight_matrix[int(s)][int(d)])

        weights = np.array(weights).astype(np.float32)

        return weights


class dataset_loader(Dataset):

    def __init__(self, x, y, w=None):
        self.x, self.y = x, y
        if w is None:
            self.w = np.ones(self.y.shape).astype(np.float32)
        else:
            self.w = w

    def __len__(self):
        return self.x.shape[0]

    def get_adj_mat(self):
        return self.y

    def get_weights(self):
        return self.w

    def set_weights(self, new_weight):
        self.w = new_weight

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]
