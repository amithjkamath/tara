# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "TARA"
copyright = "2025, Amith Kamath"
author = "Amith Kamath"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_book_theme",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

master_doc = "index"
source_suffix = [".rst", ".md"]

# -- Options for HTML output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "navigation_depth": 4,  # Depth of nested TOC items
    "show_nav_level": 1,  # Show top-level items expanded
    "collapse_navigation": True,  # Collapse subsections by default
}

# -- Options for EPUB output
epub_show_urls = "footnote"
