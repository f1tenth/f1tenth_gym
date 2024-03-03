# flake8: noqa
import os
import sys
# import f110_gym

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

# -- versions ----------------------------------------------------------------
version_match = os.environ.get("READTHEDOCS_VERSION")
# release = f110_gym.__version__
release = "1.0.0"
json_url = "https://f1tenth-gym.readthedocs.io/en/latest/_static/switcher.json"
if not version_match or version_match.isdigit() or version_match == "latest":
    if "dev" in release or "rc" in release:
        version_match = "dev"
        json_url = "_static/switcher.json"
    else:
        version_match = f"v{release}"
elif version_match == "stable":
    version_match = f"v{release}"

# -- Theme -------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_favicon = "assets/f1_stickers_02.png"
html_theme_options = {
    "logo": {
        "image_light": "src/assets/f1tenth_gym_color.svg",
        "image_dark": "src/assets/f1tenth_gym.svg",
    },
    "github_url": "https://github.com/f1tenth/f1tenth_gym",
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    }
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}
html_last_updated_fmt = "%b %d, %Y"
