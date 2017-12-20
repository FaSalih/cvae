"""Test molecule utilities."""

from pathlib import Path
import unittest
from rdkit.Chem import Mol

from ..mol_utils import load_smiles_and_data_df, smiles_to_mol, good_smiles, pad_smile

data_dir = (
    Path(__file__).parent.parent.parent / "models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)


class TestSmileChecker(unittest.TestCase):
    def test_smile_checker(self):
        self.assertIsInstance(smiles_to_mol("c1ccc1"), Mol)
        self.assertIsInstance(good_smiles("c1ccc1"), str)

        # the next 2 will log an error.
        self.assertIsNone(good_smiles("c1cc1"))
        self.assertIsNone(smiles_to_mol("c1cc1"))


class TestSmilePadding(unittest.TestCase):
    def test_smile_padding(self):
        padded_smile = pad_smile("c1ccc1", 120, "right")
        self.assertEqual(len(padded_smile), 120)
        self.assertTrue(padded_smile.endswith(" "))
        padded_smile_max = pad_smile(" " * 120, 120, "right")
        self.assertEqual(padded_smile_max, " " * 120)


class TestDataLoading(unittest.TestCase):
    def test_load_smiles_and_data_df(self):
        # run the test from VAE root dir.
        path = str(data_dir.absolute())
        cols = ["logP", "qed", "SAS"]

        s, rd, _ = load_smiles_and_data_df(path, 120, cols)  # type: ignore
        if rd is not None:
            self.assertEqual(len(s), len(rd))
            self.assertEqual(len(rd[0]), len(cols))


if __name__ == "__main__":
    unittest.main()
