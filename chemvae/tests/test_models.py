import unittest

import keras

from ..hyperparameters import params
from ..models import decoder_model, encoder_model, load_encoder


class TestEncoderLoading(unittest.TestCase):
    def test_encoder_loading(self):
        m = load_encoder(params)
        self.assertIsInstance(m, keras.Model)


# class TestDecoderLoading(unittest.TestCase):
#     def test_decoder_loading(self):
#         m = load_decoder(params)
#         m.summary()
#         self.assertIsInstance(m, keras.Model)


class TestEncoderCreation(unittest.TestCase):
    def test_model_creation(self):
        m = encoder_model(params)
        m.summary()
        self.assertIsInstance(m, keras.Model)


class TestDecoderCreation(unittest.TestCase):
    def test_model_creation(self):
        m = decoder_model(params)
        m.summary()
        self.assertIsInstance(m, keras.Model)


if __name__ == "__main__":
    unittest.main()
