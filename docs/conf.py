# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import pathlib
import sys

sys.path.insert(0, str((pathlib.Path(__file__) / '../').resolve()))


# -- Project information -----------------------------------------------------

project = 'UncertainSCI'
copyright = '2020, The Scientific Computing and Imaging Institute at the University of Utah'
author = 'Jake Bergquist, Dana Brooks, Zexin Liu, Rob MacLeod, Akil Narayan, Sumientra Rampersad, Lindsay Rupp, Jess Tate, Dan White'

# FIXME: This versioning makes no sense: we should import it from a local import of
# module itself; also need to clarify what "version" vs "release" means and adopt
# [semver](https://semver.org/).
version = '1.0'
release = '1.0.1'


# -- General configuration ---------------------------------------------------

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.intersphinx',
        'sphinx.ext.napoleon',
        'sphinx.ext.viewcode',
        'myst_nb',
        'sphinx_copybutton',
        'sphinxcontrib.bibtex',
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

master_doc = 'index'
language = 'en'


# -- Extension Configuration -------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True
autoclass_content = 'both'

# Path for bibtex files
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_encoding = 'latin'

pygments_style = 'sphinx'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
}

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
]

myst_heading_anchors = 4

nb_execution_mode = 'auto'


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
html_css_files = ['css/main.css']

html_title = project
html_logo = '_static/UncertainSCI.png'
html_theme_options = {
    'logo_only': True
}
