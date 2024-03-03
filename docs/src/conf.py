# flake8: noqa
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "f1tenth_gym"
copyright = "2023, Hongrui Zheng, Matthew O'Kelly, Aman Sinha, Joseph Auckley, Luigi Berducci, Renukanandan Tumu, Ahmad Amine"
author = "Hongrui Zheng, Matthew O'Kelly, Aman Sinha, Joseph Auckley, Luigi Berducci, Renukanandan Tumu, Ahmad Amine"

# The full version, including alpha/beta/rc tags
release = "1.0.0"
version = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"

# -- coverate test configuration -----------------------------------------------
coverage_show_missing_items = True

# -- numpydoc -----------------------------------------------------------------
numpydoc_show_class_members = False

# -- Theme -------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_favicon = "assets/f1_stickers_02.png"
html_theme_options = {
    "logo": {
        "image_light": "assets/f1tenth_gym_color.svg",
        "image_dark": "assets/f1tenth_gym.svg",
    },
    "github_url": "https://github.com/f1tenth/f1tenth_gym",
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
# html_sidebars = {
#     "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
# }
html_last_updated_fmt = "%b %d, %Y"
