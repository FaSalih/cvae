# this can be loaded because it's added to the path
# using __init__.py (check it.)
# run as python  -m scripts.mol_utils.py

from pathlib import Path
import sys

path_to_package = Path(__file__).parent.parent.resolve()
# print(path_to_package)
sys.path.append(str(path_to_package))
