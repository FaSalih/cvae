# cvae
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/ghsanti/cvae/actions/workflows/ruff.yaml/badge.svg)](https://github.com/ghsanti/cvae/actions/workflows/ruff.yaml)
[![Docs](https://github.com/ghsanti/cvae/actions/workflows/documentation.yaml/badge.svg)](https://github.com/ghsanti/cvae/actions/workflows/documentation.yaml)



Chemical Variational Autoencoder.

[Based off this great project.](https://github.com/aspuru-guzik-group/chemical_vae/)

## [API Documentation](https://ghsanti.github.io/cvae/)

## Install

* Set up env
```bash
git clone https://github.com/ghsanti/cvae cvae #clones into `cvae` folder
# replace clone by fork, if you plan to develop
cd cvae
mamba create -n cvae "python>=3.11.9,<3.12"
mamba init && source ~/.bashrc
mamba activate cvae
#check your python version now (if wrong, upgrade it.)
python --version
```
* Install dependencies
```bash
mamba install poetry graphviz -c conda-forge
mamba install nodejs -c conda-forge # only for development
# poetry will install deps and the project respecting conda.
# this also allows running the notebooks and scripts.
poetry install --without dev --sync # unless you want dev deps.
# if poetry fails run the one below
# poetry install --only-main
```

* Run a training:

You can modify the program's configuration within `scripts/train_vae.py`
```bash
python scripts/train_vae.py
```

or

```bash
python -m scripts.train_vae
```



## Updates

* Refactored. Uses Keras v3
* Removed `TerminalGRU` (this may result in less precision.)
* Supported backends: Tensorflow, Pytorch, Jax.
* Typed.
* CI.

> As long as a layer only uses APIs from the `keras.ops` namespace (or other Keras namespaces such as `keras.activations`, `keras.random`, or `keras.layers`), then it can be used with any backend – TensorFlow, JAX, or PyTorch.

By not using backend specifics, it turns the code to multibackend.

[Source: Keras.](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)


## Upcoming


0. Currently only the encoder+decoder was tested, not the property predictor. We will test it.
1. Train with more accuracy (and upload best weights.)
2. Make it run in a web browser.
3. Use it to suggest new molecules with specific properties.
4. Update notebooks

# Attributions

This is a refactor / fork of the original chemical VAE by [Aspuru-Guzik's Group](https://github.com/aspuru-guzik-group/chemical_vae/)

Part of the original readme (updated) is below (includes authors.)

------------------------


This repository contains the framework and code for constructing a variational autoencoder (VAE) for use with molecular SMILES, as described in [doi:10.1021/acscentsci.7b00572](http://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00572), with preprint at [https://arxiv.org/pdf/1610.02415.pdf](https://arxiv.org/pdf/1610.02415.pdf).

In short, molecular SMILES are encoded into a code vector representation, and can be decoded from the code representation back to molecular SMILES. The autoencoder may also be jointly trained with property prediction to help shape the latent space. The new latent space can then be optimized upon to find the molecules with the most optimized properties of interest.

In our example, we perform encoding/decoding with the ZINC dataset, and shape the latent space on prediction on logP, QED, and SAS properties.

Jupyter notebook is required to run the ipynb examples.
Make sure that the [Keras backend](https://keras.io/backend/) is set to use Tensorflow

## Components

- **scripts/train_vae.py** : main script for training variational autoencoder
- **models.py** - Library of models, contains the encoder, decoder and property prediction models.
- **tgru_k2_gpu.py** - Custom keras layer containing custom teacher forcing/sampling
- **sampled_rnn_tf.py** - Custom rnn function for tgru_k2_gpu.py, written in tensorflow backend.
- **hyperparameters/user** - Configuration.
- **mol_utils.py** - library for parsing SMILES into one-hot encoding and vice versa
- **mol_callbacks.py** - library containing callbacks used by train_vae.py
  - Includes Weight_Annealer callback, which is used to update the weight of the KL loss component
- **vae_utils.py** - utility functions for an autoencoder object, used post processing.

## Authors:
This software is written by Jennifer Wei, Benjamin Sanchez-Lengeling, Dennis Sheberla, Rafael Gomez-Bomberelli, and Alan Aspuru-Guzik (alan@aspuru.com).
It is based on the work published in https://arxiv.org/pdf/1610.02415.pdf by

 * [Rafa Gómez-Bombarelli](http://aspuru.chem.harvard.edu/rafa-gomez-bombarelli/),
 * [Jennifer Wei](http://aspuru.chem.harvard.edu/jennifer-wei),
 * [David Duvenaud](https://www.cs.toronto.edu/~duvenaud/),
 * [José Miguel Hernández-Lobato](https://jmhl.org/),
 * [Benjamín Sánchez-Lengeling](),
 * [Dennis Sheberla](https://www.sheberla.com/),
 * [Jorge Aguilera-Iparraguirre](http://aspuru.chem.harvard.edu/jorge-aguilera/),
 * [Timothy Hirzel](https://www.linkedin.com/in/t1m0thy),
 * [Ryan P. Adams](http://people.seas.harvard.edu/~rpa/'),
 * [Alán Aspuru-Guzik](http://aspuru.chem.harvard.edu/about-alan/)
