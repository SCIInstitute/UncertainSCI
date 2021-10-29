---
title: 'UncertainSCI: A Python Package fro Noninvasive Uncertainty Quantification of Simulation Pipelines'
tags:
  - Python
  - uncertainty quantification
  - computer modeling
  - polynomial choas
  - bioelectricity
authors:
  - name: Akil Narayan
    orcid: 0000-0002-5914-4207
    affiliation: "1, 2"
  - name: Jess Tate
    orcid: 0000-0002-2934-1453
    affiliation: 1
 - name: Zexin Liu
   orcid: 
   affiliation: "1, 2"
 - name: Jake A Bergquist
   orcid: 
   affiliation: "1, 3" 
 - name: Jake A Bergquist
   orcid: 
   affiliation: "1, 3" 
 - name: Sumientra Rampersad
   orcid: 
   affiliation: 4
 - name: Dan White
   affiliation: 1
 - name: Chantel Charlebois
   orcid: 
   affiliation: "1, 3" 
 - name: Lindsay Rupp
   orcid: 
   affiliation: "1, 3" 
 - name: Dana H Brooks
   affiliation: 4
 - name: Rob S MacLeod
   orcid: 
   affiliation: "1, 3" 
       
affiliations:
 - name: Scientific Computing and Imaging Institute, University of Utah, Salt Lake City, UT, USA
   index: 1
 - name: Mathematics Department, University of Utah, Salt Lake City, UT, USA
   index: 2
 - name: Biomedical Engineering Department , University of Utah, Salt Lake City, UT, USA
   index: 3
- name: Electrical and Computer Engineering Department, Northeastern University, Boston, MA, USA
   index: 4
date: July 8, 2021
bibliography: paper.bib
---

# Summary

We have developed UncertainSCI \cite{JDT:USCI} as an open source, flexible,
and easy to use tool to make modern UQ techniques more accessible in
biomedical simulation application. By implementing UncertainSCI in Python
and a noninvasive interface we were able to meet our software design goals
of 1) numerical accuracy, 2) simple application programming interface
(API), 3) adaptability to many applications and methods, and 4) interfacing
with diverse simulation software. In this paper we describe the
architecture of UncertainSCI and demonstrate its capability with ECG forward simulations utilizing  BEM and FEM, and a transcranial direct current stimulation (tDCS) examples.  

We utilized the popularity and low barrier-to-entry of Python and its
common packages to meet some of our design goals. Many software packages
and language have built in support for Python, either through hard disk
exchange or a more integrated strategy, therefore we created an interface
that is non-invasive to the modeling software and can be called in diverse
ways. Using this interface, users can initiate the UQ modeling pipeline in
UncertainSCI, pass relevant parameters to the modeling software of choice,
then pull the model solutions into UncertainSCI to compute statistics on
the uncertainty of the model.

UncertainSCI's architecture was designed to allow easy to access for users
and other software and expandable as new UQ methods are needed. This is
accomplished by having two main types of classes, for each distribution to
model and UQ method, and a series of mathematical and utility helper
libraries. The distribution classes are to facilitate characterizing data
as inputs into UQ methods like polynomial chaos expansion
(PCE)\cite{JDT:Bur2020}. Each method class has an API to add input
distributions and parameters, and to extract relevant statistics of model
uncertainty.


# Statement of need

Biomedical computer models include many input parameters and each
produces a possible error that propagates through the model. Quantification
and control of these errors through uncertainty quantification (UQ) provide
sensitivity information, a critical component when evaluating the relative
impact of parameter variation on the solution accuracy. While the need and
importance of UQ in clinical modeling is generally accepted, tools for
implementing UQ techniques remains evasive for many researchers.


# Mathematics





# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

# References

