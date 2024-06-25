import os
from pathlib import Path

from chemvae.default_config import Config
from chemvae.train_vae import main_no_prop, main_property_run

# next set of code lines are important to set path
# =================
env = os.getenv("CVAE_DATA")  # for env vars, you may not need it.
# get path to this file's dir.
dir_path = Path(env) if env else Path(__file__).parent
# Path from this dir to out data.
DATA_DIR = dir_path / ".." / "models" / "qm9"
config = Config(DATA_DIR)
# modify configuration using
# config.param = new_value
# look within `chemvae/default_config` for options.

# =================


# __name__ is set when executed from cli.
if __name__ == "__main__":
    # print("All config:", params)

    if config.do_prop_pred:
        main_property_run(config)
    else:
        print("no property pred.")
        main_no_prop(config)
