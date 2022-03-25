# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from br2.version import VERSION

# -- Project information -----------------------------------------------------

project = 'gym-softrobot'
copyright = '2022, Seung Hyun Kim, Chi-Hsien Shih'
author = 'Seung Hyun Kim, Chia-Hsien Shih'

# The full version, including alpha/beta/rc tags
release = VERSION


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    #'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'numpydoc',
    'myst_parser',
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_default_flags = ['members',  'private-members', 'special-members',  'show-inheritance']
autosectionlabel_prefix_document = True

source_parsers = {
}
source_suffix = ['.rst', '.md']

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_title = "BR2-simulator"
html_logo = "_static/assets/logo_v1.png"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/skim0119/gym-softrobot",
    "repository_branch": "main",
    "use_issues_button": True,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "logo_only": True
}
html_css_files = ['custom.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
#html_css_files = []


# -- Options for autodoc  ---------------------------------------------------
autodoc_member_order = 'bysource'

# -- Options for numpydoc ---------------------------------------------------
numpydoc_show_class_members = False
