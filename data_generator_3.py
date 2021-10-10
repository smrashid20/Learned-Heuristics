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

    def remove_node(self, x, y, exclude_node_list):

        exclusion_mask = np.zeros(self.total_num_nodes)
        for i in exclude_node_list:
            exclusion_mask[int(i)] = 1

        x_new = []
        y_new = []

        x_coord = self.x[:, :2]

        for i in range(len(x)):
            coo = x_coord[i]
            s = int(coo[0])
            d = int(coo[1])
            if exclusion_mask[s] == 0 and exclusion_mask[d] == 0:
                x_new.append(x[i])
                y_new.append(y[i])

        x_new = np.array(x_new).astype(np.float32)
        y_new = np.array(y_new).astype(np.float32)

        return x_new, y_new

    def new_training_ds(self, inclusion_node_list):

        inclusion_mask = np.zeros(self.total_num_nodes)
        for i in inclusion_node_list:
            inclusion_mask[int(i)] = 1

        x_new = []
        y_new = []

        x_coord = self.x_train_full[:, :2]

        for i in range(len(self.x_train_full)):
            coo = x_coord[i]
            s = int(coo[0])
            d = int(coo[1])
            if inclusion_mask[s] == 1 and inclusion_mask[d] == 1:
                x_new.append(self.x_train_full[i])
                y_new.append(self.y_train_full[i])

        x_new = np.array(x_new).astype(np.float32)
        y_new = np.array(y_new).astype(np.float32)

        return x_new, y_new


class dataset_loader(Dataset):

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def get_adj_mat(self):
        return self.y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
