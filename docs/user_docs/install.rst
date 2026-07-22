Installation
============

UncertainSCI supports Python 3.9, 3.10, and 3.11.

From PyPI
---------

Install the latest published package with:

.. code-block:: shell

   python -m pip install UncertainSCI

From Source
-----------

From a `source checkout <https://github.com/SCIInstitute/UncertainSCI>`_, install
the package with:

.. code-block:: shell

   python -m pip install .

Optional dependency targets are available for common tasks:

.. code-block:: shell

   python -m pip install ".[docs]"
   python -m pip install ".[interactive]"
   python -m pip install ".[dev]"

Building the notebook documentation also requires the standalone Pandoc
executable. With conda, install it with:

.. code-block:: shell

   conda install -c conda-forge pandoc

Alternatively, install Pandoc with your operating system package manager.

Targets can be combined with a comma, such as:

.. code-block:: shell
   
   python -m pip install ".[docs,interactive]"

Note that the ``dev`` target includes all optional dependencies plus development
tools such as ``flake8``, ``pytest``, and ``pytest-cov``.  If you're looking for
a quick start in development, use ``dev``.
