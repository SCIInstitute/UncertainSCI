Welcome to UncertainSCI's documentation!
========================================

.. figure:: _static/UncertainSCI.png
    :width: 495 px
    :align: center
    :target: https://www.sci.utah.edu/cibc-software/uncertainsci.html

`UncertainSCI source code <https://github.com/SCIInstitute/UncertainSCI>`_ 



About UncertainSCI
===================

UncertainSCI :cite:p:`JDT:Nar2022` is a Python-based toolkit that harnesses modern techniques to estimate model and parametric uncertainty, with a particular emphasis on needs for biomedical simulations and applications. This toolkit enables non-intrusive integration of these techniques with well-established biomedical simulation software.


Currently implemented in UncertainSCI is Polynomial Chaos Expansion (PCE) with a number of input distributions.  For more information about these techniques, see: :cite:p:`JDT:Bur2020,narayan_computation_2018,guo_weighted_2018,cohen_optimal_2017,gupta1983`.  For studies using UncertainSCI, see: :cite:p:`JDT:Ber2021,JDT:Rup2020,JDT:Rup2021,JDT:Tat2021a,JDT:Tat2021c,JDT:Tat2022`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_docs/getting_started
   user_docs/support
   tutorials/index
   dev_docs/index
   api_docs/index


Contributors
===============
Jake Bergquist, Dana Brooks, Zexin Liu, Rob MacLeod, Akil Narayan, Sumientra Rampersad, Lindsay Rupp, Jess Tate, Dan White

Acknowledgements
================

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

..
  TODO: move bibliography back here when the docutils and sphinx_rtd_theme play nicely together


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Bibliography
================
.. bibliography:: ./references.bib
  :cited:

