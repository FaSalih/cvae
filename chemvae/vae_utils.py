"""Utilities."""

import random

import numpy as np
import pandas as pd
import yaml  # should probably be just json, not yaml.

from . import mol_utils as mu
from .default_config import Config
from .models import load_decoder, load_encoder, load_property_predictor


class VAEUtils:
    """Grabs parameters, weights, chars, creates chars <=> indices dics."""

    def __init__(self, params: Config):
        """Set params."""

        self.params = params

        with open(self.params.char_file) as f:
            chars: list[str] = yaml.safe_load(f)  # json
            self.chars = chars

        self.params.NCHARS = len(self.chars)  # length of charset
        # char <=> index
        self.char_indices: dict[str, int] = dict(
            (c, i) for i, c in enumerate(self.chars)
        )
        self.indices_char: dict[int, str] = dict(
            (i, c) for i, c in enumerate(self.chars)
        )

        # encoder, decoder, predictor
        self.enc = load_encoder(self.params)  # model
        self.dec = load_decoder(self.params)  # model
        print(self.enc, self.dec)
        self.encode, self.decode = self.enc_dec_functions()
        if self.params.do_prop_pred:
            self.property_predictor = load_property_predictor(self.params)

        self.data = None  # set from file

        # Load path/name.csv without normalization as dataframe
        df, smiles_df = mu.smiles_and_full_df(
            self.params.data_file, self.params.MAX_LEN
        )
        self.smiles: list[str] = smiles_df.tolist()
        if df.shape[1] > 1:
            # set everything but the smiles
            self.data = pd.DataFrame(df.iloc[:, 1:])

        self.estimate_estandarization()
        return

    def estimate_estandarization(self):
        """After the df was split into self.smiles and self.data (un-normalised)
        This fn accesses self.data
        """
        print("Standarization: estimating mu and std values ...", end="")

        # random smiles
        smiles = self.random_molecules(n=25000)
        batch = 2500
        # hidden dim is just the latent vector dimension, a hyperparameter.
        Z = np.zeros((len(smiles), self.params.latent_dim))  # (50000,len(z))

        # [[0-2500], [2500-5000],...] iterator.
        for chunk in self.chunks(list(range(len(smiles))), batch):
            # smiles from indices
            sub_smiles = [smiles[i] for i in chunk]
            one_hot = self.smiles_to_hot(sub_smiles)  # (n_smi,max_len,nchars)
            # each row z for smile
            Z[chunk, :] = self.encode(one_hot, False)  # (50000, len(z))

        # mean for each dimension
        self.mu = np.mean(Z, axis=0)  # [1st col avg, 2nd col avg,...]
        # std for each dimension
        self.std = np.std(Z, axis=0)
        self.Z = self.standardize_z(Z)

        print("done!")
        return

    def standardize_z(self, z: np.ndarray):
        """Use the mu and std vectors to normalise z"""
        return (z - self.mu) / self.std

    def unstandardize_z(self, z: np.ndarray):
        """Use the mu and std vectors to un-normalise z"""
        return (z * self.std) + self.mu

    def perturb_z(self, z: np.ndarray, noise_norm: float, constant_norm=False):
        """Tiny deltas to the normalised z
        but z here is 25 repetitions of z for a string.
        """
        if noise_norm > 0.0:
            # n between 0 and 1
            noise_vec = np.random.normal(0, 1, size=z.shape)  # z (50000, 196)
            # F norm `sqrt(sum of the squared of all values)`
            # very small.
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:  # each smile a different noise scaling factor
                noise_amp = np.random.uniform(0, noise_norm, size=(z.shape[0], 1))
                return z + (noise_amp * noise_vec)
        else:
            return z

    def smiles_distance_z(self, smiles: str | list[str], z0):
        """smiles: are the predicted smiles
        z_rep = re-encoded & perturbed z_0 vectors (25) it's a matrix.
        z0 = I think is a single vector (unsure.)
        Returns: Distance
        """
        x = self.smiles_to_hot(smiles)
        # encodes again, adding perturbation (I think) and to 25 results.
        z_rep = self.encode(x)
        return np.linalg.norm(z0 - z_rep, axis=1)  # 'reduce' the cols

    def prep_mol_df(self, smiles: list[str], z):
        """smiles: list of smiles DECODED from perturbed versions of z
        z: single vector

        Returns: dataframe
        The df will have non duplicated smiles and extra colums with statistics.

        The way it is used is that "smiles" will be each smiles duplicated 25 times and
        perturbed, and z (z0) is the un-perturbed

        Now the smiles have 25 times more rows, I'm unsure how it's compensated.
        """
        df = pd.DataFrame({"smiles": smiles})

        # removes duplicates of perturbed-decoded string-smiles, and a count column
        # "reduce"
        sort_df = df.groupby("smiles").size().reset_index().rename(columns={0: "count"})
        # adds the count to each smile.
        df = df.merge(sort_df, on="smiles")
        df.drop_duplicates(subset="smiles", inplace=True)
        df = df[df["smiles"].apply(mu.fast_verify)]
        # len counts the rows, I guess df.shape[0] is the same
        if not df.empty:
            df["mol"] = df["smiles"].apply(mu.smiles_to_mol)
            df = df[pd.notnull(df["mol"])]

            # return error if dataframe is not empty after filtering
            if df.empty:
                print("No valid smiles found in the decoded SMILES, returning None..\nTry increasing the number of samples or reducing the noise.")
                return None
            # predicted smiles are encoded and compared initial Z
            smiles_list = df["smiles"].tolist()
            df["distance"] = self.smiles_distance_z(smiles_list, z)
            df["frequency"] = df["count"] / float(sum(df["count"]))
            df = df[["smiles", "distance", "count", "frequency", "mol"]]

            df.sort_values(by="distance", inplace=True)  # type: ignore
            df.reset_index(drop=True, inplace=True)

        # just to get the type.
        return df

    def z_to_smiles(
        self,
        z,
        decode_attempts=250,
        noise_norm=0.0,
        constant_norm=False,
        early_stop: int | None = None,
    ):
        """z: single vector
        return: decoded smiles from 25 perturbed-vector-copies
        """
        if early_stop is not None:
            Z = np.tile(z, (25, 1))  # copies all row vectors z 25 times

            # perturbed versions of the initial decoding.
            Z = self.perturb_z(Z, noise_norm, constant_norm)

            # get the character hot encoded vectors as 3D tensor
            X = self.decode(Z)  # many decodings !

            # V => character
            smiles = self.hot_to_smiles(X, strip=True)
            # remove zs decoded to same smiles and get statistics.
            df = self.prep_mol_df(smiles, z)
            if len(df) > 0:
                # iloc selects row, they then select column
                low_dist = df.iloc[0]["distance"]
                # true when closest to z is less than early stop value
                if low_dist < early_stop:
                    return df

        # same, copy Z but this time for decoding the string z.
        Z = np.tile(z, (decode_attempts, 1))
        Z = self.perturb_z(Z, noise_norm)
        X = self.decode(Z)
        smiles = self.hot_to_smiles(X, strip=True)
        df = self.prep_mol_df(smiles, z)
        return df

    def enc_dec_functions(self, standardized=True):
        """Defines "decode" fn wrt experiment parameters.

        standardized: if z for .decode(z) is standardized.
        It also stands for "standardize" for the encode(z) function.
        """
        print("Using standarized functions? {}".format(standardized))

        def decode(z, standardize=standardized):
            if standardize:
                return self.dec.predict(self.unstandardize_z(z))
            else:
                return self.dec.predict(z)

        def encode(X: np.ndarray, standardize=standardized):
            """Encodes and optionally standardizes the resulting z.

            X: hot encoded matrix of smiles.
            standardize: whether to standardise the resulting z.
            """
            z = self.enc.predict(X)[0]
            if standardize:  # in this case it's not predicting?
                return self.standardize_z(z)
            else:
                return z

        return encode, decode

    # BOTH fns use the property.predictor(z)

    # Now reports predictions after un-normalization.
    def predict_prop_Z(self, z: np.ndarray, standardized=True):
        """Predictors from z (encoded vector)"""
        if standardized:
            z = self.unstandardize_z(z)

        # both regression and logistic
        reg_prop_task = self.params.reg_prop_tasks
        logit_prop_tasks = self.params.logit_prop_tasks
        if (
            isinstance(reg_prop_task, list)
            and (len(reg_prop_task) > 0)
            and isinstance(logit_prop_tasks, list)
            and (len(logit_prop_tasks) > 0)
        ):
            reg_pred, logit_pred = self.property_predictor.predict(z)
            if isinstance(self.params.data_normalization_out_file, str):
                # un-normalise the predictions list
                # mean and std for each column of original (filtered) data.
                df_norm = pd.read_csv(self.params.data_normalization_out_file)
                reg_pred = reg_pred * df_norm["std"].values + df_norm["mean"].values
            return reg_pred, logit_pred
        # regression only scenario
        elif isinstance(reg_prop_task, list) and (len(reg_prop_task) > 0):
            reg_pred = self.property_predictor.predict(z)
            if isinstance(self.params.data_normalization_out_file, str):
                df_norm = pd.read_csv(self.params.data_normalization_out_file)
                # to watch out this flag
                reg_pred = reg_pred * df_norm["std"].values + df_norm["mean"].values
            return reg_pred
        # logit only scenario
        else:  # unsure why it encodes z here, it seems to take it as X. Is it a bug?
            # probably should just pass z to the fn below, but wait to run it.
            logit_pred = self.property_predictor.predict(self.encode(z))
            return logit_pred

    # wrapper functions
    def predict_property_function(self):
        """Similar to the previous function but starts from X (3d input tensor to encoder.)
        SO it encodes and predicts.
        """

        # Now reports predictions after un-normalization.
        def predict_prop(X):
            # both regression and logistic
            reg_prop_task = self.params.reg_prop_tasks
            logit_prop_tasks = self.params.logit_prop_tasks
            if (
                isinstance(reg_prop_task, list)
                and (len(reg_prop_task) > 0)
                and isinstance(logit_prop_tasks, list)
                and (len(logit_prop_tasks) > 0)
            ):
                reg_pred, logit_pred = self.property_predictor.predict(self.encode(X))
                if isinstance(self.params.data_normalization_out_file, str):
                    df_norm = pd.read_csv(self.params.data_normalization_out_file)
                    reg_pred = reg_pred * df_norm["std"].values + df_norm["mean"].values
                return reg_pred, logit_pred
            # regression only scenario
            elif isinstance(reg_prop_task, list) and (len(reg_prop_task) > 0):
                reg_pred = self.property_predictor.predict(self.encode(X))
                if isinstance(self.params.data_normalization_out_file, str):
                    df_norm = pd.read_csv(self.params.data_normalization_out_file)
                    # same, to watch out reg_pred
                    reg_pred = reg_pred * df_norm["std"].values + df_norm["mean"].values
                return reg_pred

            # logit only scenario
            else:
                logit_pred = self.property_predictor.predict(self.encode(X))
                return logit_pred

        return predict_prop

    def ls_sampler_w_prop(self, size=None, batch=2500, return_smiles=False):
        """Returns encoded (predictions) smiles Z and corresponding data for them."""
        if self.data is None:
            print("use this sampler only for external property files")
            return

        cols = []
        reg_prop_tasks = self.params.reg_prop_tasks
        logit_prop_tasks = self.params.logit_prop_tasks
        if isinstance(reg_prop_tasks, list):
            cols += reg_prop_tasks
        if isinstance(logit_prop_tasks, list):
            cols += logit_prop_tasks

        # data is the table without smiles column
        # note that it must be in the same order than smiles.
        idxs = (
            self.random_idxs(size)
            if isinstance(size, "int")
            else list(range(len(self.smiles)))
        )
        # grab random smiles
        smiles = [self.smiles[idx] for idx in idxs]
        # grab corresponding data
        data = [self.data.iloc[idx] for idx in idxs]
        # hidden dimensions for all input smiles
        Z = np.zeros((len(smiles), self.params.latent_dim))

        # this avoids having more than 1 cube in memory at a time.
        for chunk in self.chunks(list(range(len(smiles))), batch):
            sub_smiles = [smiles[i] for i in chunk]
            # basically a small_X (3D)
            one_hot = self.smiles_to_hot(sub_smiles)
            # will slowly populate the predictions.
            Z[chunk, :] = self.encode(one_hot)

        if return_smiles:
            return Z, data, smiles

        return Z, data

    def smiles_to_hot(
        self, smiles: str | list[str], canonize_smiles=False, check_smiles=False
    ):
        """charIndex=Indices_to_char_dict[pass_char]
        Make [0,...,1,...] with 1 at char index.

        If check_smiles=True it returns list of 'bad smiles' instead.

        Returns: 3D np.ndarray of hotencoded vectors
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        if canonize_smiles:
            smiles = [mu.canon_smiles(s) for s in smiles]

        if check_smiles:
            smiles = mu.smiles_to_hot_filter(smiles, self.char_indices)

        z = mu.smiles_to_hot(
            smiles,
            self.params.MAX_LEN,
            self.params.PADDING,
            self.char_indices,
            self.params.NCHARS,
        )
        return z

    Tensor3D = list[list[list[float]]]

    def hot_to_smiles(self, hot_x: Tensor3D, strip=False):
        """From 3D tensor of Probabilities for each character & smile
        to smiles flat list.
        """
        smiles = mu.hot_to_smiles(hot_x, self.indices_char)
        if strip:  # postprocessing
            smiles = [s.strip() for s in smiles]
        return smiles

    # I removed the n=None, this may trigger some errors if they used it.
    def random_idxs(self, n: int):
        """n: how many random indices to return in the new list"""
        return random.sample([i for i in range(len(self.smiles))], n)

    def random_molecules(self, n: int):
        """n: how many random smiles strings to return in the new list"""
        print(f"Sampling {n} random molecules from the dataset..")
        print(f"Total number of molecules in the dataset: {len(self.smiles)}")
        return random.sample(self.smiles, n)

    @staticmethod
    def chunks(indices: list[int], n: int):
        """Yield successive n-sized chunks from l.
        indices: list of indices
        n: number of elements, for a slice of n elements.
        """
        for i in range(0, len(indices), n):
            yield indices[i : i + n]
