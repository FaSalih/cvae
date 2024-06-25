import shutil
from pathlib import Path

import mkdocs.plugins


@mkdocs.plugins.event_priority(-50)
def on_startup(command, dirty):
    """Makes the markdown files.

    Prepares Markdown files from Python file,
    for mkdocs to create HTML files from there.
    """
    this_dir = Path(__file__).parent  # get this directory

    python_file_paths = this_dir.glob("./chemvae/**/*.py")  # glob our files

    shutil.copy("./README.md", "docs/index.md")  # use the readme as intro page.

    for python_file_path in python_file_paths:
        if not python_file_path.is_file:
            continue

        py_rel_path = python_file_path.relative_to(this_dir)
        if py_rel_path.match("*test_*") or py_rel_path.name.endswith("__init__.py"):
            continue

        dot_path = str(py_rel_path).replace("/", ".")
        module_path = str(Path(dot_path).with_suffix(""))
        md_file = (dot_path + ".md").replace("chemvae.", "docs/")
        with open(md_file, "w") as f:
            f.write(f"::: {str(module_path)}")
