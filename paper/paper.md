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

We have developed UncertainSCI [@USCI] as an open source tool designed to make modern uncertatinty quantification (UQ) techniques more accessible in biomedical simulation applications. UncertainSCI is implemented in Python with a noninvasive interface to meet our software design goals of 1) numerical accuracy, 2) simple application programming interface (API), 3) adaptability to many applications and methods, and 4) interfacing with diverse simulation software.  Using a Python implementation in UncertainSCI allowed us to utilize the popularity and low barrier-to-entry of Python and its common packages and to leverage the built in integration and support for Python in common simulation software packages and languages. Additionally, we used non-invasive UQ techniques and created a similarly non-invasive interface to external modeling software that can be called in diverse ways, depending on the complexity and level of Python integration of the external simulation pipeline. We have developed and included examples applying UQ to relatively simple 1D simulations implemented in python and to bioelectric field simulations implemented in external software packages to demonstrate the use of UncertainSCI and the effectiveness of the archetecture and implementation in achieving our design goals.

# Statement of need

Biomedical computer models include many input parameters and variation of each can propagate through a model to produce a subsequently varying output. Quantification and control of these errors through UQ provides statistics and sensitivity information, a critical component when evaluating the relative impact of parameter variation on the solution accuracy. While the need and importance of UQ in clinical modeling is generally accepted, tools for implementing UQ techniques remains evasive for many researchers.


# Mathematics

In UncertainSCI, we quantify forward parametric uncertainty in cardiac simulations using polynomial Chaos expansions (PCE) [@ACN:Xiu2010]. PCE attempts to approximate the dependence of a quantity of interest (QoI) from the forward simulation on a finite number of random parameters via a multivariate polynomial function of those parameters. The polynomial function constitutes an emulator, from which statistics of the QoI, including the mean, variance, and parameter sensitivities, are efficiently computed from the polynomial. UncertainSCI uses  a non-intrusive strategy to construct this polynomial dependence through least-squares approximation, where data for the least-squares problem is collected through an ensemble of simulations of the forward model. 

The efficiency of PCE to comput UQ of a forward simulation depends on efficient selection of parameter samples, or experimental design.  UncetainSCI strategically samples the parameter space  the procedure of Weighted Approximate Fekete Points (WAFP) [ACN:Guo2018,@JDT:Burk:2020], which computes a geometrically unstructured parametric design of experiments as a special type of weighted D-optimal (determinant-maximizing) design.  More precisely, the design is computed through a greedy algorithm that iteratively adds parametric samples that maximize a weighted matrix determinant.  The maximization is computed over a discrete candidate set compute by random sampling from a specially tailored probability that exploits a concentration of measurable phenomena to provably increase the quality of the candidate set [ACN:Coh2017].  Sampling from this induced distribution for independent parameters is computationally efficient, having complexity that is linear in the number of parameters [ACN:Nar2018].

Once the experimental design is created through the WAFP procedure with induced distribution sampling, an ensemble of forward simulations is collected from the simulation software, and UncertainSCI produces a PCE emulator through a (weighted) least-squares procedure. From this least-squares procedure, UncertainSCI also can compute residuals and cross-validation metrics, and can adaptively tune the expressivity of the PCE emulator based on a userprescribed tolerance and/or computational budget.

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

