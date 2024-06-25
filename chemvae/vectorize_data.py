from .mol_utils import load_smiles_and_data_df, smiles_to_hot
from typing import Literal, overload
from .default_config import Config
import numpy as np


@overload
def vectorize_data(
    config: Config, do_prop_pred: Literal[True]
) -> tuple[np.ndarray, np.ndarray, list, list]: ...


@overload
def vectorize_data(
    config: Config, do_prop_pred: Literal[False]
) -> tuple[np.ndarray, np.ndarray]: ...


def vectorize_data(
    config: Config, do_prop_pred: bool = True
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list, list]:
    """Split the dataframe to: smiles_tensor[, prediction data]

    Returns:
      If true: (smiles_train, smiles_val, [reg_tasks_train, val], [log_tasks_t, v] )
      If false: (smiles_train, smiles_val)

    The smiles are hot-encoded to a 3-D tensor.
    It makes sure that it's multiple of batch size, and data is randomly selected.
    """
    MAX_LEN = config.MAX_LEN
    CHARS = config.CHARS
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))

    ## Load data for properties
    if do_prop_pred and config.data_normalization_out_file:
        normalize_out = config.data_normalization_out_file
    else:
        normalize_out = None

    ################ matches columns in csv file ###########
    reg_props = config.reg_prop_tasks  # list of names
    logit_props = config.logit_prop_tasks

    if do_prop_pred and not reg_props and not logit_props:
        raise ValueError("please especify logit and/or reg tasks")

    # here we get the csv-data split, and optionally a "normed write out" for reg data.
    smiles, Y_reg, Y_logit = load_smiles_and_data_df(
        csv_file_path=config.data_file,  # smiles + data
        max_len=MAX_LEN,
        reg_tasks=reg_props,
        logit_tasks=logit_props,
        normalize_out=normalize_out,  # path to write normalised reg data out|None
    )

    # subset of data if needed.
    if config.limit_data:
        # sample indices within the range. The number collected is limit_data value.
        sample_idx = np.random.choice(
            np.arange(len(smiles)), config.limit_data, replace=False
        )
        smiles = list(np.array(smiles)[sample_idx])  # sublist of size config.limit_data
        if config.do_prop_pred and config.data_file:
            if Y_reg:
                Y_reg = Y_reg[sample_idx]  # basically the rows of the original DF.
            if Y_logit:
                Y_logit = Y_logit[sample_idx]

    print("Training set size is", len(smiles))
    print("total chars:", NCHARS)

    X = smiles_to_hot(smiles, MAX_LEN, config.PADDING, CHAR_INDICES, NCHARS)
    # if less than the batch size the `//` gives 0.
    if X.shape[0] % config.batch_size != 0:
        # make it multiple of batch_size, discard excedent.
        to_length = X.shape[0] // config.batch_size * config.batch_size
        X = X[:to_length]
        if config.do_prop_pred:
            if Y_reg:
                Y_reg = Y_reg[:to_length]
            if Y_logit:
                Y_logit = Y_logit[:to_length]

    np.random.seed(config.RAND_SEED)
    rand_idx = np.arange(X.shape[0])
    np.random.shuffle(rand_idx)  # shuffles the rows' indices.

    TRAIN_FRAC = 1 - config.val_split
    num_train = int(X.shape[0] * TRAIN_FRAC)

    # or gets 0
    if num_train % config.batch_size != 0:
        num_train = num_train // config.batch_size * config.batch_size
        print("num_train ", num_train)

    # makes the indices for each
    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train) :]

    if config.test_idx_file:
        np.save(config.test_idx_file, test_idx)

    # grab the rows with an list of random indices.
    X_train, X_test = X[train_idx], X[test_idx]
    print(f"shape of training input vector : {X_train.shape}")

    if do_prop_pred:
        # !# add Y_train and Y_test here
        Y_train = []
        Y_test = []
        if Y_reg:
            Y_reg_train, Y_reg_test = Y_reg[train_idx], Y_reg[test_idx]
            Y_train.append(Y_reg_train)
            Y_test.append(Y_reg_test)
        if Y_logit:
            Y_logit_train, Y_logit_test = Y_logit[train_idx], Y_logit[test_idx]
            Y_train.append(Y_logit_train)
            Y_test.append(Y_logit_test)

        return X_train, X_test, Y_train, Y_test

    else:
        return X_train, X_test
