# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'PortfolioQtOpt'
copyright = '2022, Eneko Osaba, Guillaume Gelabert'
author = 'Eneko Osaba, Guillaume Gelabert'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxcontrib.bibtex",
]

# Latex config
mathjax3_config = {'chtml': {
    "scale": 1,                      # global scaling factor for all expressions
    "minScale": .5,                  # smallest scaling factor to use
    "matchFontHeight": True,         # true to match ex-height of surrounding font
    "mtextInheritFont": False,       # true to make mtext elements use surrounding font
    "merrorInheritFont": True,       # true to make merror text use surrounding font
    "mathmlSpacing": False,          # true for MathML spacing rules, false for TeX rules
    "skipAttributes": {},            # RFDa and other attributes NOT to copy to the output
    "exFactor": .5,                  # default size of ex in em units
    "displayAlign": 'right',        # default for indentalign when set to 'auto'
    "displayIndent": '0'             # default for indentshift when set to 'auto'
  },
};

# Add bibliography file name
bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Using notebooks in other format.
# https://docs.readthedocs.io/en/stable/guides/jupyter.html
nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}