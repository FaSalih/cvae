import os
from pathlib import Path

from chemvae.default_config import Config
from chemvae.train_vae import main_no_prop, main_property_run

# next set of code lines are important to set path
# =================

if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "tensorflow"

env = os.getenv("CVAE_DATA")
dir_path = Path(env) if env else Path(__file__).parent
DATA_DIR = dir_path / ".." / "models" / "qm9"
config = Config(DATA_DIR)

# =================

if __name__ == "__main__":
    if config.do_prop_pred:
        main_property_run(config)
    else:
        print("\n Training withough property predictor.\n")
        main_no_prop(config)
