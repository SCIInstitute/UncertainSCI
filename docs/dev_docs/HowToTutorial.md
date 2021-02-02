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

**This tutorial demonstrates how to use markdown to create new tutorials for UcertainSCI.  It will walk through all the files needed and the basic structure needed expected for tutorials.  Knowledge of Markdown, Github, Github pages, and Python will facilitate **
#### Software requirements
##### UncertainSCI
To make a Tutorial for UncertainSCI, start with an up-to-date version of the code and documentation.  Download the source code or clone the repository from [github](https://github.com/SCIInstitute/UncertainSCI.git).  We suggest [creating a fork](#creating-your-uncertainsci-fork) of the repository so that you can track your changes and create pull requests to the UncertainSCI repository.

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
The main file needed for a new tutorial is a markdown file.  The file should have an file ending of *.md* and should be located in the `UncertainSCI/docs/user_docs/` directory.

#### Added Figures

#### Additional Files

#### Linking to New Tutorial


### Testing Documentation
**This chapter describes how to...**
**Scope: [Testing Documentation](#Testing-Documentation) **
#### Installing Jekyll

#### Building Documentation

#### Other Strategies


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



### Supplemental Materials

#### Example Scripts
#### Movies
#### Datasets


