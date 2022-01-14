# Getting Started with UncertainSCI


## System Requirements

Requires Python 3 and modules listed in `requirements.txt`

## Getting UncertainSCI

The easyiest way to get UncertainSCI is to use pip.  Just run `pip install UncertainSCI` or `python -m pip install UncertainSCI` and pip will download and install the latest version of UncertainSCI.  This will also try to download and install the relevent dependencies too.  

To get pip, see its [documentation](https://pip.pypa.io/en/stable/installation/).  

The source code for UncertainSCI can be downloaded from the [Github page](https://github.com/SCIInstitute/UncertainSCI).   

## Installing UncertainSCI From Source 

To build UncertainSCI from a downloaded copy of the source code, ensure that the proper dependencies are biult. This can be done by navigating to the source code directory and using the command `pip install -r requirements.txt` or `python -m pip install -r requirements.txt` in the terminal window.  The dependencies can also be installed manually with pip or from their source distributions.  

With the dependencies installed, UncertainSCI can be built by calling the `setup.py` file in python with `python setup.py build` to only build locally or `python setup.py install` to install UncertainSCI with the python distribution. 

Alternatively, UncertainSCI can be isntalled locally with the command  `pip install .` or  `python -m pip install .`.  This will only build 

The UncertainSCI source directory will need to be added to the PYTHONPATH environment variable for it to be called from other locations.  More on adding directories to the PYTHONPATH can be found [here](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html).  




## Using UncertainSCI

Check out the [tutorials](../tutorials/index.html) and the [API documentation](../api_docs/index.html). 

`demos/build_pce.py`  is particularly useful for a quick demonstration of parametric uncertainty quantification using polynomial chaos methods.



