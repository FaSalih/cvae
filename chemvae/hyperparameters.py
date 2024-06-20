"""Make a class with the configuration.

1. takes UserConfig class from ./config
2. retrieves a Config class after some processing.
3. Using classes instead of objects gives more type info.
"""

from pathlib import Path

import yaml

# configs
from .config.user_config import UserConfig

# this dir
dir_path = Path(__file__).parent

# data relative to here.
DATA_DIR = dir_path / ".." / "models" / "qm9"


class Config(UserConfig):
    """Updated configuration used thoughout the project."""

    def __init__(self, data_dir: Path):
        """Set and update properties.

        DATA_DIR: where the configuration files are.
        """
        super().__init__()

        # updates paths.
        for k, v in list(vars(self).items()):
            if k.endswith("file") and isinstance(v, str):
                setattr(self, k, str(data_dir.joinpath(v).resolve()))
            elif k.endswith("path"):
                setattr(self, k, str(data_dir.joinpath(v).resolve()))
        # load chars and set them

        self.CHARS = yaml.safe_load(open(self.char_file))
        self.NCHARS = len(self.CHARS)

    def print_params(self):
        """Pretty print params."""
        print("Using hyper-parameters:")
        for key, value in vars(self).items():
            print("{:25s} - {:12}".format(key, str(value)))
        print("rest of parameters are set as default")


params = Config(DATA_DIR)
