from importlib.metadata import PackageNotFoundError, version as package_version


project = "gym-softrobot"
author = "Seung Hyun Kim, Chia-Hsien Shih"
copyright = "2022–2026, Seung Hyun Kim and Chia-Hsien Shih"

try:
    release = package_version("gym-softrobot")
except PackageNotFoundError:
    release = "development"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "myst_parser",
]

autosectionlabel_prefix_document = True
exclude_patterns = ["_build", "adr"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]

html_theme = "celshast"
html_title = "gym-softrobot"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/skim0119/gym-softrobot/",
    "source_branch": "main",
    "source_directory": "docs/",
}
