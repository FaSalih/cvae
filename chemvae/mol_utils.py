"""Convert and check smiles. Load and write dataframe."""

import logging
import pickle as pkl

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import Mol

# started adding
from typing_extensions import Literal

logging.getLogger("autoencoder")
logging.getLogger().setLevel(20)
logging.getLogger().addHandler(logging.StreamHandler())

# =================
# text io functions
# ==================


def smiles_to_mol(smiles: str) -> None | Mol:
    """Check that it can be converted to Mol.

    Otherwise catches the error and returns None
    Returns: None|Mol
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    # except Exception as ex:
    except:
        # print(f"can not convert {smiles}", ex)
        print(f"can not convert {smiles}")
        pass
    return None


def verify_smiles(smile: str):
    """Check that smile is not falsy."""
    return (
        (smile != "") and pd.notnull(smile) and (Chem.MolFromSmiles(smile) is not None)
    )


def good_smiles(smile: str):
    """Verify smile and return either canonical or none.

    Returns: smile or None.
    """
    if verify_smiles(smile):  # not falsy
        return canon_smiles(smile)
    else:
        return None


Padding_type = Literal["right", "left", "none"]


def pad_smile(string: str, max_len: int, padding: Padding_type = "right") -> str:
    """For "string length > max" raise exception.

    Return: smile padded to the max length
    """
    to_add = max_len - len(string)
    if is_valid_len(string, max_len):
        if padding == "right":
            return string + " " * to_add
        elif padding == "left":
            return " " * to_add
        elif padding == "none":
            return string
        else:
            raise Exception("padding must be left|right|none.")
    else:
        raise Exception("Smile was too long.")


def filter_valid_length(strings: list[str], max_len: int):
    """Return strings with length <= the max length."""
    return [s for s in strings if len(s) <= max_len]


def filter_valid_smiles_return_invalid(strings: list[str], max_len: int):
    """Filter strings with length above max_length.

    Returns
    -------
    1. A list of valid strings (list[str])
    2. A list of indices of invalid strings. (list[int])

    """
    filter_list: list[int] = []
    new_smiles: list[str] = []
    for idx, s in enumerate(strings):
        if len(s) > max_len:
            filter_list.append(idx)
        else:
            new_smiles.append(s)
    return new_smiles, filter_list


def smiles_to_hot(
    smiles: list[str],
    max_len: int,
    padding: Padding_type,
    char_indices: dict[str, int],
    nchars: int,
):
    """Populate binary tensor (smiles, max_len, nchars)."""
    smiles = [
        pad_smile(i, max_len, padding) for i in smiles if is_valid_len(i, max_len)
    ]
    # nchars=n of components of each hot-enc vector (n of != strings available in dataset.)
    # nice tensor
    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)

    # populate it
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                # for example "c" has some char index in that dictionary.
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                # if the lookup fails
                print("ERROR: Check chars file. Bad SMILES: ", smile, "char n: ", char)
                raise e
    return X


def smiles_to_hot_filter(smiles: list[str], char_indices: dict[str, int]):
    """Put smiles with "non-supported" characters into a list."""
    filtered_smiles: list[str] = []
    for smile in smiles:
        for char in smile:
            try:
                char_indices[char]
            except KeyError:
                break
        else:
            filtered_smiles.append(smile)
    return filtered_smiles


def term_hot_to_smiles(
    hot_x: list[np.ndarray], temperature: float, indices_chars: dict[int, str]
):
    """Pass the logits through softmax (more or less.).

    hot_x = hot encoded smile
    Return: smile string
    """
    temp_string = ""
    for j in hot_x:
        # j=character as probs vector
        # get max index after softmax (more or less)
        index = thermal_argmax(j, temperature)
        # map to string.
        temp_string += indices_chars[index]
    return temp_string


Tensor3D = list[list[list[float]]]


def hot_to_smiles(hot_x: Tensor3D, indices_chars: dict[int, str]):
    """Return: smile string from argmax(hot_smile)"""
    smiles: list[str] = []
    for x in hot_x:  # go over rows (smiles)
        temp_str = ""
        for j in x:  # go over characters, as Probs arrays
            index = np.argmax(j)  # get max index
            temp_str += indices_chars[index]
        smiles.append(temp_str)
    return smiles


def thermal_argmax(prob_arr: np.ndarray, temperature: float):
    """Add a little bit of probabilistic behaviour to the chosen string."""
    prob_arr = np.log(prob_arr) / temperature
    prob_arr = np.exp(prob_arr) / np.sum(np.exp(prob_arr))
    print(prob_arr)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn(
            "Probabilities to sample add to more than 1, {}".format(prob_arr.sum())
        )
        prob_arr = prob_arr / (prob_arr.sum() + 0.0000000001)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn("Probabilities to sample still add to more than 1")
    # imagine a dice where each face has a P, this is what they do here.
    # it returns a 1 in the selected component of the array of same length.
    return np.argmax(np.random.multinomial(1, prob_arr, 1))


def load_smiles(smi_file, max_len: int | None = None, return_filtered=False):
    """1. Get smiles under max length from file.

    2. Opt: Pad to max_len, if max_len is specified
    3. Opt: get the indices of bad smiles
    Returns: nice_smiles_list [, bad_smiles_indices_list ]
    """
    if smi_file[-4:] == ".pkl":  # binary file
        with open(smi_file, "rb") as f:
            smiles = pkl.load(f)  # visible outside the scope as well
    else:  # assume file is a text file
        with open(smi_file, "r") as f:
            smiles = f.readlines()
        smiles = [i.strip() for i in smiles]

    if max_len is not None:
        if return_filtered:
            smiles, filtrate = filter_valid_smiles_return_invalid(smiles, max_len)
            if len(filtrate) > 0:
                print("Filtered out {} smiles above max_len".format(len(filtrate)))
            return smiles, filtrate

        else:
            old_len = len(smiles)
            smiles = filter_valid_length(smiles, max_len)
            diff_len = old_len - len(smiles)
            if diff_len != 0:
                print("Filtered out {} smiles above max_len".format(diff_len))

    return smiles


Column_names = list[str] | None


def load_smiles_and_data_df(
    csv_file_path: str,
    max_len: int,
    reg_tasks: list[str] | None = None,
    logit_tasks: list[str] | None = None,
    normalize_out: str | None = None,
    dtype: str = "float64",
):
    """1. loads csv as df from path and filters using max_len.

    2. Splits into data_df_filtered, smiles_series_filtered
    3. `smiles_series.tolist()`
    4. Computes properties of the data_df_filtered.
    5. Writes out the statistics for the selected columns (if normalise_out!=None)

    Arguments:
    ---------
    csv_file_path: path to data file
    max_len: filters long smiles
    reg_tasks : [colnames] that correspond to regression tasks.
    logit_tasks : [same] for logit tasks
    normalize_out: path to write out normalized reg cols (only) csv or None.

    """
    if logit_tasks is None:
        logit_tasks = []
    if reg_tasks is None:
        reg_tasks = []

    # filters only on max_len, splits dataframe
    df, smiles_series = smiles_and_full_df(csv_file_path, max_len)
    smiles: list[str] = smiles_series.tolist()

    reg_data_df = df[reg_tasks]  # subsets of columns
    logit_data_df = df[logit_tasks]  # subsets of columns

    # normalise regression data in df
    if len(reg_tasks) != 0 and normalize_out is not None:
        df_norm = pd.DataFrame()
        # stats for each column.
        df_norm["mean"] = reg_data_df.mean(axis=0)
        df_norm["std"] = reg_data_df.std(axis=0)
        reg_data_df = (reg_data_df - df_norm["mean"]) / df_norm["std"]
        # write out normalised colums csv.
        df_norm.to_csv(normalize_out)  # this will be loaded by the sampler !

    # return new arrays
    if len(logit_tasks) != 0 and len(reg_tasks) != 0:
        # .values returns the table as an array.
        # I removed the np.vstack(reg_data_df.values)
        return (
            smiles,
            reg_data_df.values.astype(dtype),
            logit_data_df.values.astype(dtype),
        )
    elif len(reg_tasks) != 0:
        return smiles, reg_data_df.values.astype(dtype), None
    elif len(logit_tasks) != 0:
        return smiles, logit_data_df.values.astype(dtype), None
    else:
        return smiles, None, None


# ==============================================
# ---------------make charset------------
# ==============================================
def make_charset_list_from_list(smi_list):
    """Create a list of unique characters.

    Does not hot encode the chars.
    """
    char_lists = [list(smi) for smi in smi_list]
    chars = list(set([char for sub_list in char_lists for char in sub_list]))
    chars.append(" ")

    return chars


def make_charset_file_from_file(smi_file, char_file):
    with open(smi_file, "r") as afile:
        unique_chars = set(afile.read())
    bad = ["\n", '"']
    unique_chars = [c for c in unique_chars if c not in bad]
    unique_chars.append(" ")
    print("found {} unique chars".format(len(unique_chars)))
    # why this replacement? could be due to JSON but idk.
    astr = str(unique_chars).replace("'", '"')
    print(astr)

    with open(char_file, "w") as afile:
        afile.write(astr)
    print("wrote charset to {}".format(char_file))
    return


def smiles_and_full_df(csv_file_path: str, max_len: int):
    """Split df, smiles.

    smiles are <= max_len & and stripped out.
    """
    # I guess smiles are within the pandas dataframe.
    df = pd.read_csv(csv_file_path)

    # strip strings
    # df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    smi_col_name = df.columns[0]
    df[smi_col_name] = df[smi_col_name].str.strip()

    # select good smiles
    # df = df[df.iloc[:, 0].str.len() <= max_len]
    df = df[df[smi_col_name].str.len() <= max_len]
    smiles_df = df[smi_col_name]

    return df, smiles_df


# =================
# data parsing io functions
# ==================


def is_valid_len(smile: str, max_len: int):
    return len(smile) <= max_len


def CheckSmiFeasible(smi: str):
    # try smi=>mol=>smi
    # otherwise return false
    try:
        get_molecule_smi(Chem.MolFromSmiles(smi))
    except Exception as ex:
        print(f"Can't convert {smi}", ex)
        return False
    return True


def balanced_parentheses(smiles: str):
    s = []

    balanced = True
    index = 0
    while index < len(smiles) and balanced:
        token = smiles[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()

        index += 1

    return balanced and len(s) == 0


def matched_ring(s: str):
    """Check that rings are represented correctly.

    Example: c1cccccc1 i.e (c1's close the ring.)
    c: aromatic, C: aliphatic
    """
    return s.count("1") % 2 == 0 and s.count("2") % 2 == 0


def fast_verify(s: str):
    """Ring and balanced parenthesis check."""
    return matched_ring(s) and balanced_parentheses(s)


def get_molecule_smi(mol_obj) -> str:
    return Chem.MolToSmiles(mol_obj)


def canon_smiles(smi: str) -> str:
    """Smile to canonical smile.

    Doesn't rdkit have a canonicalise smile fn?
    """
    return Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True
    )
