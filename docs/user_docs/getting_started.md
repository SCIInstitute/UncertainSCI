# Getting Started with UncertainSCI


## System Requirements

Requires Python 3 and modules listed in `requirements.txt`

## Getting UncertainSCI

The easyiest way to get UncertainSCI is to use pip.  Just run `pip install UncertainSCI` or `python -m pip install UncertainSCI` and pip will download and install the latest version of UncertainSCI.  This will also try to download and install the relevent dependencies too.  

To get pip, see its [documentation](https://pip.pypa.io/en/stable/installation/).  

The source code for UncertainSCI can be downloaded from the [Github page](https://github.com/SCIInstitute/UncertainSCI).   

### Installing UncertainSCI From Source 

UncertainSCI can be built from source code using the `setup.py` script.  To call this script, navagate to the source root directory in a terminal window and run the command `pip install .` or `python -m pip install .`  

Alternatively, you can call the script with `python setup.py install`.  pip is still called to install the dependencies, but they could be installed manually instead.

## Running UncertainSCI Demos

There are a number of demos included with UncertainSCI to test it's installation and to demonstrate its use.  

`demos/build_pce.py`  is particularly useful for a quick demonstration of parametric uncertainty quantification using polynomial chaos methods.  To run this demo, make sure that UncertainSCI is [installed](#getting-uncertainsci), then simply call the script with python using the command `python demo/build_pce.py`.  Other demos can be run similarily.  

We have included a number of demos and [tutorials](../tutorials/index.html) to teach users how to use UncertainSCI with your own applications.  The [API documentation](../api_docs/index.html) explains the implementation of UncertainSCI in more detail. 
