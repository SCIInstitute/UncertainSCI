---
title: Making Tutorials
category: developer documentation
tags: tutorial, contribute
layout: default_toc
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
<link rel="stylesheet" href="css/main.css">

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering
(U24EB029012) from the National Institutes of Health.

&nbsp;

Authors:  
Jess Tate 

### Contents

*  [Overview](#overview)
	-  [Software Requirements](#software-requirements)
*  [Files Needed for a New Tutorials](#files-needed-for-a-new-tutorials)
	- [Overview of Files Needed for each Module](#overview-of-files-needed-for-each-module)
	- [2.2 Module Configuration File](#22-module-configuration-file)
	- [2.3 Module Source Code](#23-module-source-code)
		+ [2.3.1 Module Header File](#231-module-header-file)
		+ [2.3.2 Module Code File](#232-module-code-file)
	- [2.4 Module UI Code](#24-module-ui-code)
		+ [2.4.1 Module Design File](#241-module-design-file)
		+ [2.4.2 Module Dialog Header](#242-module-dialog-header)
		+ [2.4.3 Module Dialog Code](#243-module-dialog-code)
	- [2.5 Algorithm Code](#25-algorithm-code)
		+ [2.5.1 Module Algorithm Header](#251-module-algorithm-header)
		+ [2.5.2 Module Algorithm Code](#252-module-algorithm-code)
*  [3 Example: Simple Module Without UI](#example-simple-module-without-ui)
	- [3.1 Module Config File](#31-module-config-file)
	- [3.2 Module Header File](#32-module-header-file)
	- [3.3 Module Source Code](#33-module-source-code)
	- [3.4 Building and Testing](#34-building-and-testing)
*  [8 Documenting the New Module](#documenting-the-new-module) -->


### Overview

**This tutorial demonstrates how to use markdown to create new tutorials for UcertainSCI.  It will walk through all the files needed and the basic structure needed expected for tutorials.  Knowledge of Markdown, Github, Github pages, and Python will be useful. If you have questions, [please ask](https://github.com/SCIInstitute/UncertainSCI/discussions).**

#### Software requirements
##### UncertainSCI
To make a Tutorial for UncertainSCI, start with an up-to-date version of the code and documentation.  Download the source code or clone the repository from [github](https://github.com/SCIInstitute/UncertainSCI.git).  We suggest [creating a fork](#creating-your-uncertainsci-fork) of the repository so that you can track your changes and create pull requests to the UncertainSCI repository.  UncertainSCI requirements are found [here](../specs.html)

##### Dependencies and Development Tools
UncertainSCI uses Github Pages to host tutorial documentation.   This platform converts markdown files to html for webview using Jekyll.  Testing the new documentation may require building the web pages locally for viewing.  Instructions for this process can be found on the [Github Help pages](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll).

##### Creating Your UncertainSCI Fork
With your own github account, go to the [UncertainSCI github page](https://github.com/SCIInstitute/UncertainSCI). Click the fork button on the upper right side of the page. It will ask you where to move the fork to, chose your own account. Once the repository is forked, clone it to your local machine with the following command.  

```
$git clone https://github.com/[yourgithubaccount]/UncertainSCI.git
```

After the the code is cloned, navigate to the repository directory and add the upstream path to the original UncertainSCI repository.  

```
$git remote add upstream https://github.com/SCIInstitute/UncertainSCI.git
```

You should be able to see both your and the original repositories when you use the command:     

```
$git remote -v
```

The fork is good to go, but you will need to sync the fork occasionally to keep up with the changes in the main repository.  To sync your fork, use the following commands:

```
$git fetch upstream

$git checkout master

$git merge upstream/master
```
You should sync and merge your fork before you start a new module and before you create a pull request.  
It is a good practice to create a new branch in your fork for every module you will be adding.  The command to create a new branch is:

```
$git checkout -b [branch_name]
```

Please see the [github help page](https://help.github.com) for more information.



### Files Needed for a New Tutorial

**This chapter will describe the files need to create a Tutorial for UncertainSCI.**

**Scope: [Overview of Files Needed for a Tutorial](#overview-of-files-needed-for-a-tutorial)  - [Markdown file](#markdown-file) - [Added Figures](#added-figures) - [Additional Files](#additional-files) - [Linking to New Tutorial](#linking-to-new-tutorial)

#### Overview of Files Needed for a Tutorial
To make a new tutorial, a markdown file is required for the content of the tutorial.  Other files, such as images, may also be included.  In addtion to the new files for the tutorial, a link to the new tutorial should be added to the *User Documents* file.  

#### Markdown File
The main file needed for a new tutorial is a markdown file.  The file should have an file ending of *.md* and should be located in the `UncertainSCI/docs/user_docs/` directory.  There is a [template file](../user_doc/template.md) that can be used, or an existing tutorial like this one can be used.  Markdown files must have a header and should look something like:
```
---
title: Tutorial Title
category: user documentation
tags: tutorial, etc.
layout: default_toc
---
```
The file should also include a call to the style sheet soon after the header:
```
<link rel="stylesheet" href="css/main.css">
```
If using math equations in a latex, a call to the mathjax server is required at this point too:
```
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
```
That is all that is needed to make the file appear properly.  Next start with the content of the tutorial, starting with the acknowledgement and the list of authors.
```
This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

Authors:  
Smart people here
```
A table of contents can be added at the beginning of the document, yet there should also be an interactive menu on the sidebar.
```
### Contents

*  [Overview](#overview)
        -  [Software Requirements](#software-requirements)
*  [Chapter Name](#chapter-name)
        - [Overview of Chapter](#overview-of-chapter)
        - [Section](#section)
            - [Subsection](#section)
```
A mini table of contents can be added to the begining of each section.  The one for this section looks like this:
```
**Scope: [Overview of Files Needed for a Tutorial](#overview-of-files-needed-for-a-tutorial)  - [Markdown file](#markdown-file) - [Added Figures](#added-figures) - [Additional Files](#additional-files) - [Linking to New Tutorial](#linking-to-new-tutorial)
```
For examples on how to added different content, such as figures, equations, etc., refer to the [content section](#adding-content) as well as the [Content Guide](#content-guide).

#### Added Figures

Most tutorials require a number of screenshots and other images.  Figures and images should be added in a folder for each tutorial in the  `UncertainSCI/docs/user_docs/` folder. The figure folder should be named after the tutorial, for example, the images in this tutorial are in a folder called `HowToTutorial_figures`.  Instructions on how to use the images and figures are found [here](#figures)

#### Additional Files
Additional files added to the `user_docs` folder should be minimized as much as possible.  Example scripts should be located in the `UncertainSCI/examples` directory and example data will, generally, need seperate hosting. However, if one or two files are needed, they may be added to the  `UncertainSCI/docs/user_docs/` with a consistent naming scheme.  Bibtex file with a matching name should be added in  `UncertainSCI/docs/user_docs/`, yet if multiple additional files are needed, they should placed in a new folder indicating the tutorial: `UncertainSCI/docs/user_docs/[tutorial name]_files`.  

#### Linking to New Tutorial

For the new tutorial to be visible on the [user documentation page](../user.html), add a link to the [`UncertainSCI/docs/user.md`](https://github.com/SCIInstitute/UncertainSCI/blob/master/docs/user.md) file.  

### Testing Documentation

**This chapter describes how to test the look and content of the new tutorial.  Test the  generated github-pages with either a local jekyll build or using the online build on a fork of UncertainSCI.**

**Scope: [Testing Documentation](#Testing-Documentation) - [Installing Jekyll](#installing-jekyll) - [Building Documentation](#building-documentation) - [Testing on a Fork](#testing on a fork)**

#### Installing  Jekyll

#### Building Documentation

#### Testing on a Fork


### Adding Content

**In this chapter...**

**Scope: [Duplicate the Previous Module](#41-duplicate-the-previous-module) - [Creating a Custom UI](#42-creating-a-custom-ui) - [Connecting UI to the Module](#43-connecting-ui-to-the-module) - [Adding an Input Port](#44-adding-an-input-port) - [Finished Code](#45-finished-code)**

#### Figures
#### Math
#### Citations
#### Snippets
#### Links
#### Referencing Sphynx


### Content Guide

TODO

### Supplemental Materials

TODO

#### Example Scripts
#### Movies
#### Datasets


