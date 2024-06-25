import shutil
from pathlib import Path

import mkdocs.plugins


@mkdocs.plugins.event_priority(-50)
def on_startup(command, dirty):
    """Makes the markdown files.

    Removes docs if present, and uses modules to create files"""
    this_dir = Path(__file__).parent

    filenames = this_dir.glob("./chemvae/**/*.py")
    index = Path("docs/index.md")
    shutil.copy("./README.md", index)

    with index.open("a") as index:
        for file_path in filenames:
            if not file_path.is_file:
                continue

            file = file_path.relative_to(this_dir)
            if file.match("*test_*") or file.name.endswith("__init__.py"):
                continue

            with_dots = str(file).replace("/", ".")
            module_path = Path(with_dots).with_suffix("")
            md_file = str(Path(with_dots).with_suffix(".md")).replace(
                "chemvae.", "docs/"
            )
            with open(md_file, "w") as f:
                f.write(f"::: {str(module_path)}")
