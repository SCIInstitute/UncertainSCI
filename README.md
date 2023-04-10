# UncertainSCI


A Python-based toolkit that harnesses modern techniques to estimate model and parametric uncertainty, with a particular emphasis on needs for biomedical simulations and applications. This toolkit enables non-intrusive integration of these techniques with well-established biomedical simulation software.

![UncertainSCI](docs/_static/UncertainSCI.png "UncertainSCI")


![All Builds](https://github.com/SCIInstitute/UncertainSCI/workflows/Build/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/SCIInstitute/UncertainSCI/badge.svg?branch=master)](https://coveralls.io/github/SCIInstitute/UncertainSCI?branch=master)
[![status](https://joss.theoj.org/papers/660d2fe53fbf67dd2714e9546251bd33/status.svg)](https://joss.theoj.org/papers/660d2fe53fbf67dd2714e9546251bd33)

## Overview

UncertainSCI is an open-source tool designed to make modern uncertainty quantification (UQ) techniques more accessible in biomedical simulation applications.   UncertainSCI uses noninvasive UQ techniques, specifically polynomial Chaos estimation (PCE), with a similarly noninvasive interface to external modeling software that can be called in diverse ways.  PCE and UncertainSCI allows users to propagate the effect of input uncertainty on model results, providing essential context for model stability and confidence needed in many modeling fields.  Users can run UncertainSCI by setting input distributions for a model parameters, setting up PCE, sampling the parameter space, running the samples sets within the target model, and compiling output statistics based on PCE.  This process is breifly describe in the [getting started guide](../user_docs/getting_started.html#quick-guide), and more fully explained in the [API documentation](../api_docs/index.html), and supplied [demos and tutorials](../tutorials/index.html).

## Publications



## Documentation

<https://uncertainsci.readthedocs.io>

## Getting Started Guide

<https://uncertainsci.readthedocs.io/en/latest/user_docs/getting_started.html>

## License

Distributed under the MIT license. See ``LICENSE`` for more information.

## Acknowledgements


This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.
