---
title: 'UncertainSCI: A Python Package for Noninvasive Parametric Uncertainty Quantification of Simulation Pipelines'
tags:
  - Python
  - uncertainty quantification
  - computer modeling
  - polynomial chaos
  - bioelectricity
authors:
  - name: Jess Tate
    orcid: 0000-0002-2934-1453
    affiliation: 1
  - name: Zexin Liu
    orcid: 0000-0003-3409-5709
    affiliation: "1,2" 
  - name: Jake A Bergquist
    orcid: 0000-0002-4586-6911
    affiliation: "1, 3, 4" 
  - name: Sumientra Rampersad
    orcid: 0000-0001-9860-4459
    affiliation: "5, 6"
  - name: Dan White
    affiliation: 1
  - name: Chantel Charlebois
    orcid: 0000-0002-4139-3539
    affiliation: "1, 3" 
  - name: Lindsay Rupp
    orcid: 0000-0002-2688-7688
    affiliation: "1, 3, 4" 
  - name: Dana H Brooks
    orcid: 0000-0003-3231-6715
    affiliation: 6
  - name: Rob S MacLeod
    orcid: 0000-0002-0000-0356
    affiliation: "1, 3, 4"
  - name: Akil Narayan
    orcid: 0000-0002-5914-4207
    affiliation: "1, 2"
affiliations:
 - name: Scientific Computing and Imaging Institute, University of Utah, Salt Lake City, UT, USA
   index: 1
 - name: Mathematics Department, University of Utah, Salt Lake City, UT, USA
   index: 2
 - name: Biomedical Engineering Department , University of Utah, Salt Lake City, UT, USA
   index: 3
 - name: Nora Eccles Cardiovascular Research and Training Institute, University of Utah, Salt Lake City, UT, USA
   index: 4
 - name: Physics Department, University of Massachusetts, Boston, MA, USA
   index: 5
 - name: Electrical and Computer Engineering Department, Northeastern University, Boston, MA, USA
   index: 6
date: Mar 29, 2023
bibliography: paper.bib
---

# Summary

We have developed UncertainSCI [@USCI] as an open-source tool designed to make modern uncertainty quantification (UQ) techniques more accessible in biomedical simulation applications. UncertainSCI is implemented in Python with a noninvasive interface to meet our software design goals of 1) numerical accuracy, 2) simple application programming interface (API), 3) adaptability to many applications and methods, and 4) interfacing with diverse simulation software.  Using a Python implementation in UncertainSCI allowed us to utilize the popularity and low barrier-to-entry of Python and its common packages and to leverage the built-in integration and support for Python in common simulation software packages and languages. Additionally, we used noninvasive UQ techniques and created a similarly noninvasive interface to external modeling software that can be called in diverse ways, depending on the complexity and level of Python integration in the external simulation pipeline. We have developed and included examples applying UncertainSCI to relatively simple 1D simulations implemented in Python, and to bioelectric field simulations implemented in external software packages, which demonstrate the use of UncertainSCI and the effectiveness of the architecture and implementation in achieving our design goals.  UnceratainSCI differs from similar software, notably [UQLab](https://www.uqlab.com), [Uncertainpy](https://uncertainpy.readthedocs.io/en/latest/), and [Simnibs](https://simnibs.github.io/simnibs/build/html/tutorial/advanced/uq.html), in that it can be efficiently and non-invasively used with external simulation software, specifically with high resolution 3D simulations often used in Bioelectric field simulations.  \autoref{fig:pipeline} illustrates the use of UncertainSCI in computing UQ with modeling pipelines for bioelectricity simulations.  


![User pipeline for UncertainSCI.  After the user inputs parameter distributions, UncertainSCI will compute an efficient sampling scheme.  The parameter samples are run through the targeted modeling pipeline, which can be implemented in external software tools.  The computed solutions are collected and compiled into relevant statistics with UncertainSCI. \label{fig:pipeline}](UncertainSCI_pipeline.png){ width=100% }


# Statement of need

Biomedical computer models include many input parameters that do not have precisely defined values, for example because their value defines physiological processes that are not uniform across patients. As such, any simulation output necessarily has some uncertainty associated with the uncertain value of the input parameter. Exploration and quantification of this model output uncertainty is challenging when more than a single parameter is present; biomedical computer models often have 5-20 such parameters. Quantification of this uncertainty through UQ techniques provides statistics and sensitivity information, a critical component when evaluating the relative impact of parameter variation on the solution accuracy.  While the need and importance of UQ in clinical modeling is generally accepted, automated tools for implementing UQ techniques remain evasive for many researchers.   UncertainSCI has been used to quantify uncertainty in multiple modeling pipelines, including:  cardiac tissue modeling [@JAB:Rup2020; @JDT:Rup2021; @JDT:Ber2021], electrocardiographic (ECG) simulation [@RSM:Swe2011; @RSM:Gen2005b], ECG imaging (ECGI) [@JDT:Tat2021a], transcranial current stimulation (tCS) modeling [@JDT:Tat2021c; @JDT:Ram2021], and electrocorticography (ECoG) stimulation [@JDT:Tat2021c; @JDT:Ram2021].  


# Mathematics

In UncertainSCI, we quantify forward parametric uncertainty in cardiac simulations using polynomial chaos expansions (PCE) [@ACN:Xiu2010]. Although we also have an implementation of Monte Carlo sampling, and we intend to expand to UncertianSCI to include other UQ methods, such as Guassian process emulators [@ACN:Ras2004], Markov Chain Monte Carlo [@JDT:Rah2016] and Bayesian inference[@JDT:Xu2014], we primarily focused on PCE due to its non-invasive formulation and computational efficiency.  PCE approximates the dependence of a quantity of interest (QoI) that is the model output from the forward simulation on a finite number of random parameters via a multivariate polynomial function of those parameters. With $u$ the QoI (scalar-valued for simplicity), and $p \in \mathbb{R}^d$ the vector of uncertain parameters, the PCE approach builds the function $u_N$, given as,
\begin{align}
  u(p) \approx u_N(p) = \sum_{j=1}^N c_j \phi_j(p),
\end{align}
where the $\{\phi_j\}_{j=1}^N$ functions are multivariate polynomials, and the coefficients $\{c_j\}_{j=1}^N$ are learned through an ensemble of collected data,
\begin{align}\label{eq:training}
  \left\{ p_j\right\}_{j=1}^M \xrightarrow[\textrm{Forward model $u$}]{} \left\{ u(p_j) \right\}_{j=1}^M \xrightarrow[\textrm{Emulator training}]{} \left\{c_j\right\}_{j=1}^N.
\end{align}
Typically one seeks convergence of $u_N$ to $u$ in an $L^2$-type norm weighted by the probability density of the parameters $p$. The polynomial function $u_N$ constitutes an emulator for QoI $u$, from which statistics of the QoI, including the mean, variance, and parameter sensitivities, are efficiently computed from the polynomial. UncertainSCI uses a particular type of pseudo-random sampling for the parameter design $\{p_j\}_{j=1}^M$, along with a particular weighted least-squares approach for emulator training, both of which are recently developed advances in high-dimensional approximation. The entire procedure \eqref{eq:training} is non-intrusive, since the forward model need only be queried at a particular set of samples.

The efficiency of PCE for analysis of UQ in a forward simulation depends on efficient selection of parameter samples $\{p_j\}_{j=1}^M$.  The goal is to use as few samples $M$ as possible while still ensuring that $u_N$ is an accurate emulator for $u$. UncertainSCI uses a two-step approach to strategically sample the parameter space:
1.  A discrete candidate set is generated via random sampling with respect to a ''biased'' probability measure $\mu$, that is distinct from (but related to) the probability distribution of the parameter $p$.
2. A weighted D-optimal design is sought by subsampling this discrete candidate set. UncertainSCI uses a greedy approach to produce an approximation to such a design.

The probability measure $\mu$ that must be randomly sampled is a distribution that exploits a concentration of measurable phenomena to provably increase the quality of the candidate set [@ACN:Coh2017].  Sampling from this distribution when the components of the parameter vector $p$ are independent is computationally efficient, having complexity that is linear in the number of parameters $d$ [@ACN:Nar2018]. The relatively large candidate set generated from this random sampling is pruned via subsampling using a weighted D-optimal design optimization. UncertainSCI's algorithm for this approach approximately computes a weighted D-optimal design via the weighted approximate Fekete points (WAFP) procedure [@ACN:Guo2018; @JDT:Bur2020], which greedily maximizes a weighted matrix determinant. The result is a geometrically unstructured parameter design of $M$ samples for use in the pipeline \eqref{eq:training}.

Once the experimental design is created through the WAFP procedure, an ensemble of forward simulations $\left\{ u(p_j) \right\}_{j=1}^M$ is collected from the simulation software, and UncertainSCI produces a PCE emulator $u_N$ through a (weighted) least-squares procedure. From this emulator, UncertainSCI can compute statistics, sensitivities, residuals, and cross-validation metrics, and can adaptively tune the complexity of the PCE emulator based on a user-prescribed tolerance and/or computational budget.

# Acknowledgements

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

# References

