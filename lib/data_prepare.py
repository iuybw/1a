import pickle
import torch
import numpy as np
import os
import gc
from .utils import print_log, StandardScaler, vrange
from .adjacent_matrix_norm import load_adj
# ! X shape: (B, T, N, C)

def load_pkl(pickle_file: str) -> object:
    """
    Load data from a pickle file.

    Args:
        pickle_file (str): Path to the pickle file.

    Returns:
        object: Loaded object from the pickle file.
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"Unable to load data from {pickle_file}: {e}")
        raise
    return pickle_data

def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, batch_size=64, log=None, train_size=0.6
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )
    if train_size != 0.6:
        drop_last=True
    else:
        drop_last=False
    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )

    return trainset_loader, valset_loader, testset_loader, scaler

def get_dataloaders_from_index_data_MTS(
    data_dir,
    in_steps=12,
    out_steps=12,
    tod=False,
    dow=False,
    y_tod=False,
    y_dow=False,
    batch_size=64,
    log=None,
):
    data = np.load(os.path.join(data_dir, f"data.npz"))["data"].astype(np.float32)
    index = np.load(os.path.join(data_dir, f"index_{in_steps}_{out_steps}.npz"))

    x_features = [0]
    if tod:
        x_features.append(1)
    if dow:
        x_features.append(2)

    y_features = [0]
    if y_tod:
        y_features.append(1)
    if y_dow:
        y_features.append(2)

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    # Parallel
    # x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    # y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    # x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    # y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    # x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    # y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    # x_train = data[x_train_index][..., x_features]
    # y_train = data[y_train_index][..., y_features]
    # x_val = data[x_val_index][..., x_features]
    # y_val = data[y_val_index][..., y_features]
    # x_test = data[x_test_index][..., x_features]
    # y_test = data[y_test_index][..., y_features]

    # Iterative
    x_train = np.stack([data[idx[0] : idx[1]] for idx in train_index])[..., x_features]
    y_train = np.stack([data[idx[1] : idx[2]] for idx in train_index])[..., y_features]
    x_val = np.stack([data[idx[0] : idx[1]] for idx in val_index])[..., x_features]
    y_val = np.stack([data[idx[1] : idx[2]] for idx in val_index])[..., y_features]
    x_test = np.stack([data[idx[0] : idx[1]] for idx in test_index])[..., x_features]
    y_test = np.stack([data[idx[1] : idx[2]] for idx in test_index])[..., y_features]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler

def get_dataloaders_from_index_data_Test(
    data_dir,
    in_steps=12,
    out_steps=12,
    tod=False,
    dow=False,
    y_tod=False,
    y_dow=False,
    batch_size=64,
    log=None,
):
    data = np.load(os.path.join(data_dir, f"data.npz"))["data"].astype(np.float32)
    index = np.load(os.path.join(data_dir, f"index_{in_steps}_{out_steps}.npz"))

    x_features = [0]
    if tod:
        x_features.append(1)
    if dow:
        x_features.append(2)

    y_features = [0]
    if y_tod:
        y_features.append(1)
    if y_dow:
        y_features.append(2)

    train_index = index["train"]  # (num_samples, 3)
    # val_index = index["val"]
    test_index = index["test"]

    # Parallel
    # x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    # y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    # x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    # y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    # x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    # y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    # x_train = data[x_train_index][..., x_features]
    # y_train = data[y_train_index][..., y_features]
    # x_val = data[x_val_index][..., x_features]
    # y_val = data[y_val_index][..., y_features]
    # x_test = data[x_test_index][..., x_features]
    # y_test = data[y_test_index][..., y_features]

    # Iterative
    x_train = np.stack([data[idx[0] : idx[1]] for idx in train_index])[..., x_features]
    # y_train = np.stack([data[idx[1] : idx[2]] for idx in train_index])[..., y_features]
    # x_val = np.stack([data[idx[0] : idx[1]] for idx in val_index])[..., x_features]
    # y_val = np.stack([data[idx[1] : idx[2]] for idx in val_index])[..., y_features]
    x_test = np.stack([data[idx[0] : idx[1]] for idx in test_index])[..., x_features]
    y_test = np.stack([data[idx[1] : idx[2]] for idx in test_index])[..., y_features]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    # x_train[..., 0] = scaler.transform(x_train[..., 0])
    # x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    # print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    # print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    # trainset = torch.utils.data.TensorDataset(
    #     torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    # )
    # valset = torch.utils.data.TensorDataset(
    #     torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    # )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    # trainset_loader = torch.utils.data.DataLoader(
    #     trainset, batch_size=batch_size, shuffle=True
    # )
    # valset_loader = torch.utils.data.DataLoader(
    #     valset, batch_size=batch_size, shuffle=False
    # )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return testset_loader, scaler