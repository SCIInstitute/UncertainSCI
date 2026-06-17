# UncertainSCI Python Environment
People that install UncertainSCI can be sensibly categorized as follows:
- **Users**: people who want to use UncertainSCI but do not need to meddle with its internals
- **Developers**: people who want to meddle with UncertainSCI's internals

These instructions explain how to install UncertainSCI for both of these use cases.

**Python Versions:** UncertainSCI officially supports Python versions 3.9 to 3.11.  Newer Python may incidentally work, but at this time its support has not been tested.

## Dependency Overview
This is an overview of the dependencies for UncertainSCI for various use cases.

### Base Dependencies
At a minimum, UncertainSCI requires all packages in the project's requirements.txt.

### Doc Dependencies
To build the docs, UncertainSCI requires
- pandoc
- sphinx
- sphinxcontrib-bibtex
- sphinx-markdown-tables
- nbsphinx
- recommonmark

### Testing & Linting Dependencies
To test and lint the codebase, UncertainSCI requires
- flake8
- pytest
- pytest-cov

### Nice-to-haves
- ipython
- jupyter

## Environment for Production
It is assumed your production environment does not need docs (e.g., pandoc), testing nor linting (e.g., flake8 nor pytest), nor interactive interfaces (e.g., jupyter).  The packages and tools for these features are therefore omitted from these installation instructions.

### With conda
1. Change to the repository root.
2. Set the `ENVNAME` and `PYVER` variables.  On *NIX,
    ```shell
    export ENVNAME=<your environment name here>
    export PYVER=<Python version between 3.9 and 3.11>
    ```
    On Windows,
    ```shell
    $ENVNAME="<your environment name here>"
    $PYVER="<Python version between 3.9 and 3.11>"
    ```
3. Create and activate the conda environment, then install UncertainSCI,
    ```shell
    conda create -n $ENVNAME python=$PYVER pandoc
    conda activate $ENVNAME
    python -m pip install .
    ```

### Without conda
1. You have already installed the version of Python with which you wish to run UncertainSCI.
2. Change to the repository root.
3. Create and activate a virtual environment inside of which to run UncertainSCI,
    ```shell
    python -m venv .venv
    ```
    Activate the virtual environment according to your system.
4. Install UncertainSCI,
    ```shell
    python -m pip install .
    ```

## Environment for Development
It is assumed to develop UncertainSCI that you want to be able to build the docs, test and flake the codebase, and have reasonably modern interactive tools.

### With conda
1. Change to the repository root.
2. Set the `ENVNAME` and `PYVER` variables.  On *NIX,
    ```shell
    export ENVNAME=<your environment name here>
    export PYVER=<Python version between 3.9 and 3.11>
    ```
    On Windows,
    ```shell
    $ENVNAME="<your environment name here>"
    $PYVER="<Python version between 3.9 and 3.11>"
    ```
3. Create and activate the conda environment, then install UncertainSCI,
    ```shell
    conda create -n $ENVNAME python=$PYVER pandoc
    conda activate $ENVNAME
    python -m pip install .
    ```
4. Install development helpers,
    ```shell
    python -m pip install pandoc sphinx sphinxcontrib-bibtex sphinx-markdown-tables nbsphinx recommonmark  # docs
    python -m pip install flake8 pytest pytest-cov  # linting and testing
    python -m pip install ipython jupyter  # nice-to-haves
    ```

### Without conda
1. You have already installed the version of Python with which you wish to run UncertainSCI.
2. Change to the repository root.
3. Create and activate a virtual environment inside of which to run UncertainSCI,
    ```shell
    python -m venv .venv
    ```
    Activate the virtual environment according to your system.
4. Install UncertainSCI,
    ```shell
    python -m pip install .
    ```
5. Install development helpers,
    ```shell
    python -m pip install pandoc sphinx sphinxcontrib-bibtex sphinx-markdown-tables nbsphinx recommonmark  # docs
    python -m pip install flake8 pytest pytest-cov  # linting and testing
    python -m pip install ipython jupyter  # nice-to-haves
    ```
