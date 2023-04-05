# Template for Tutorial

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
<link rel="stylesheet" href="_static/css/main.css">

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

Authors:  
Smart people here


## Overview

** Overview of the document **

### Software Requirements

#### Overview Subsection


## Chapter Name

** Overview text for the Chapter.  In this case there will be examples of several types of content**

### Section
Example section with subsection.  

Use as many paragraphs as needed.  

There are many markdown guides.  Here are some examples:
<https://www.markdownguide.org/basic-syntax/>
<https://guides.github.com/features/mastering-markdown/>

#### Subsection
Example Subsection.  These can go to 6 `#`'s.  Subsections are optional for  table of contents and chapter scope.

### Figures

![Example for including an image in tutorial.](../_static/UncertainSCI.png "UncertainSCI example image")

### Math
Math equations use [MathJax](http://docs.mathjax.org/en/latest/basic/mathematics.html). This requires the inclusion of this near the beginning of the document: 
```
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
```
Example equation:
\\[ x = {-b \pm \sqrt{b^2-4ac} \over 2a} \\]
$$ \frac{\partial \rho}{\partial t} + \nabla \cdot \vec{j} = 0 \,. \label{eq:continuity} $$

inline equations use the `\\(\mathbf{p}\\)` sytanx: \\(\mathbf{p}\\) 

### Citations

Sphinx has a built in citation manager for bibtex: [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/). Works well for RST, but we are still working on it for markdown.  The sphinxcontrib-bibtex is built to run with rst in Sphinx. However, it can be used in markdown using the [AutoStructify](https://recommonmark.readthedocs.io/en/latest/auto_structify.html) package.

```eval_rst
The whole paragraph will need to be in the eval_rst block :cite:p:`JDT:Bur2020`. For multiple references: :cite:p:`JDT:Bur2020,gupta1983`
```


add a bibliography section
````
```eval_rst
.. bibliography::
```
````

### Snippets
Inline snippets `like this`.  Muliple lines:
```
# # Define model
N = int(1e2)  # Number of degrees of freedom of model
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)
```

### Links

Internal link: [Overview](#overview)

External link: <https://www.markdownguide.org>, or [Markdown](https://www.markdownguide.org)

### Tables

Tables can be used with normal markdown syntax with the [sphinx-markdown-tables](https://github.com/ryanfox/sphinx-markdown-tables) package

```
| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |
```

| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |


### Referencing Sphynx


To link the UncertainSCI API generated using Sphynx, Use this syntax: [`[text](../api_docs/pce.html#polynomial-chaos-expansions)`](../api_docs/pce.html#polynomial-chaos-expansions)







            
