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

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.


Authors:  
Jess Tate 

### Contents

*  [Overview](#overview)
	-  [Software Requirements](#software-requirements)
*  [Files Needed for a New Tutorial](#files-needed-for-a-new-tutorial)
	-  [Overview of Files Needed for a Tutorial](#overview-of-files-needed-for-a-tutorial)
	-  [Markdown File](#markdown-file)
	-  [Added Figures](#added-figures)
       -  [Additional Files](#additional-files)
       -  [Linking to New Tutorial](#linking-to-new-tutorial)
*  [Testing Documentation](#testing-documentation)
       - [Testing Locally](#testing-locally)
            + [Installing Jekyll](#installing-jekyll)
            + [Building Documentation](#building-documentation)
       - [Testing on a Fork](#testing-on-a-fork)
* [Adding Content](#adding-content)
       - [Figures](#figures)
       - [Math](#math)
       - [Citations](#citations)
       - [Snippets](#snippets)
       - [Links](#links)
       -[Referencing Sphynx](#referencing-sphynx)
* [Content Guide](#content-guide)
* [Supplemental Materials](#supplemental-materials)
       - [Example Scripts](#example-scripts)
       - [Movies](#movies)
       - [Datasets](#datasets)

### Overview

**This tutorial demonstrates how to use markdown to create new tutorials for UcertainSCI.  It will walk through all the files needed and the basic structure needed expected for tutorials.  Knowledge of Markdown, Github, Github pages, and Python will be useful. If you have questions, [please ask](https://github.com/SCIInstitute/UncertainSCI/discussions).**

#### Software requirements
##### UncertainSCI
To make a Tutorial for UncertainSCI, start with an up-to-date version of the code and documentation.  Download the source code or clone the repository from [github](https://github.com/SCIInstitute/UncertainSCI.git).  We suggest [creating a fork](#creating-your-uncertainsci-fork) of the repository so that you can track your changes and create pull requests to the UncertainSCI repository.  UncertainSCI requirements are found [here](../specs.html)

##### Dependencies and Development Tools
UncertainSCI uses Github Pages to host tutorial documentation.   This platform converts markdown files to html for web viewing using Jekyll.  Testing the new documentation may require building the web pages locally for viewing.  This will require installing Ruby, from whcih Bundler and Jekyll can be installed.  Ruby can be installed through [many channels](https://www.ruby-lang.org/en/documentation/installation/).  Ruby helps control the appropriate versions of the additional dependencies, including Jekyll and Bundler.  See the [testing section](#installing-jekyll) for detailed instructions.   

##### Creating Your UncertainSCI Fork
With your own github account, go to the [UncertainSCI Github page](https://github.com/SCIInstitute/UncertainSCI). Click the fork button on the upper right side of the page. It will ask you where to move the fork to, chose your own account. Once the repository is forked, clone it to your local machine with the following command.  

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

Please see the [Github help page](https://help.github.com) for more information.



### Files Needed for a New Tutorial

**This chapter will describe the files need to create a Tutorial for UncertainSCI.**

**Scope: [Overview of Files Needed for a Tutorial](#overview-of-files-needed-for-a-tutorial)  - [Markdown file](#markdown-file) - [Added Figures](#added-figures) - [Additional Files](#additional-files) - [Linking to New Tutorial](#linking-to-new-tutorial)**

#### Overview of Files Needed for a Tutorial
To make a new tutorial, a markdown file is required for the content of the tutorial.  Other files, such as images, may also be included.  In addition to the new files for the tutorial, a link to the new tutorial should be added to the *User Documents* file.  

#### Markdown File
The main file needed for a new tutorial is a markdown file.  The file should have an file ending of *.md* and should be located in the `UncertainSCI/docs/user_docs/` directory.  There is a [template file](https://github.com/SCIInstitute/UncertainSCI/blob/master/docs/user_docs/template.md) that can be used, or an existing tutorial like this one can be used.  Markdown files must have a header and should look something like:
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
A mini table of contents can be added to the beginning of each section.  The one for this section looks like this:
```
**Scope: [Overview of Files Needed for a Tutorial](#overview-of-files-needed-for-a-tutorial)  - [Markdown file](#markdown-file) - [Added Figures](#added-figures) - [Additional Files](#additional-files) - [Linking to New Tutorial](#linking-to-new-tutorial)
```
For examples on how to added different content, such as figures, equations, etc., refer to the [content section](#adding-content) as well as the [Content Guide](#content-guide).

#### Added Figures

Most tutorials require a number of screenshots and other images.  Figures and images should be added in a folder for each tutorial in the  `UncertainSCI/docs/user_docs/` folder. The figure folder should be named after the tutorial, for example, the images in this tutorial are in a folder called `HowToTutorial_figures`.  Instructions on how to use the images and figures are found [here](#figures)

#### Additional Files
Additional files added to the `user_docs` folder should be minimized as much as possible.  Example scripts should be located in the `UncertainSCI/examples` directory and example data will, generally, need separate hosting. However, if one or two files are needed, they may be added to the  `UncertainSCI/docs/user_docs/` with a consistent naming scheme.  Bibtex file with a matching name should be added in  `UncertainSCI/docs/user_docs/`, yet if multiple additional files are needed, they should placed in a new folder indicating the tutorial: `UncertainSCI/docs/user_docs/[tutorial name]_files`.  

#### Linking to New Tutorial

For the new tutorial to be visible on the [user documentation page](../user.html), add a link to the [`UncertainSCI/docs/user.md`](https://github.com/SCIInstitute/UncertainSCI/blob/master/docs/user.md) file.  

### Testing Documentation

**This chapter describes how to test the look and content of the new tutorial.  Test the  generated github-pages with either a local Jekyll build or using the online build on a fork of UncertainSCI.**

**Scope: [Testing Documentation](#Testing-Documentation) - [Testing Locally](#testing-locally) - [Installing Jekyll](#installing-jekyll) - [Building Documentation](#building-documentation) - [Testing on a Fork](#testing-on-a-fork)**

#### Testing Locally
Testing the documentation locally involves building and running a jekyll server with the documentation on your local machine.  These instructions are adapted from [Github's help page](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll).

##### Installing Jekyll
To install the Jekyll, make sure that [Ruby is installed](https://www.ruby-lang.org/en/documentation/installation/).  We will use Ruby to install [Bundler](https://bundler.io) and [Jekyll](https://jekyllrb.com/docs/).  In a terminal window, enter the following command:
```
gem install bundler
```
Next, navigate to the docs folder of your local clone of the UncertainSCI repo.  
```
cd UncertainSCI/docs
```
You can use bundler to install Jekyll and all the subsequent dependencies to run the local Jekyll server.  Just run:
```
bundle install
```
This will install the dependencies specified in the `Gemfile` and `Gemfile.lock` files. This should be all that that is required to run a jekyll server locally to test the documentation pages. It is worth noting that `bundle install` should also reinstall the dependencies if they happen to change between local builds, yet the other steps do not need to be repeated as often.   Please [ask](https://github.com/SCIInstitute/UncertainSCI/discussions) if you have any questions

##### Building Documentation
Once Ruby, Bunlder, Jekyll, and all the dependencies are [installed properly](#installing-jekyll), the Jekyll server can be launched with the following command:
```
bundle exec jekyll serve
```
You should see a sequence of text indicating that the server is running, similar to:
```
Configuration file: /Users/test/UncertainSCI/docs/_config.yml
            Source: .
       Destination: ./_site
 Incremental build: disabled. Enable with --incremental
      Generating... 
                    done in 0.505 seconds.
 Auto-regeneration: enabled for '.'
    Server address: http://127.0.0.1:9001/UncertainSCI/
  Server running... press ctrl-c to stop.
  ```
This indicates that the server built correctly and the documentation page can be view by entering the server address into a browser, in this case `http://127.0.0.1:9001/UncertainSCI/`.  You should be able to see the UncertainSCI docs page, from which you can navigate to the new tutorial via the browser links.  The address of the new tutorial page will be the same as the location of the Markdown file, with the `.md` ending replaced with `.html`, e.g., `http://127.0.0.1:9001/UncertainSCI/user_docs/[tutorial_name].html`.  The server should regenerate pages when markdown files are changed, allowing you to view the changes quickly by refreshing the browser. For more information, see the [Github help page](https://docs.github.com/en/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll)

If the server does not start correctly, and you see errors in the output after the command `bundle exec jekyll serve`, it is likely that some of the dependencies were not installed correctly, particularly if the error is `bundler: failed to load command: jekyll...`.  First, run `bundle install` again.  if the error persists, you can try [updating your Ruby gems](https://rubygems.org/pages/download):
```
gem update --system
```
Or by running:
```
bundle update
```
One final check would be the versions of Ruby and Bundler that are running, which may also need upgrading.  Please [ask](https://github.com/SCIInstitute/UncertainSCI/discussions) if you have questions.  

#### Testing on a Fork
In addition to building and testing the documentation pages locally, they can also be tested and viewed using a [fork of the UncertainSCI repo](#creating-your-uncertainsci-fork).  Once the fork is created, you will be able to see the documentation page for master branch of the fork by visiting the page `http://[yourgithubaccount].github.io/UncertainSCI/`.  Changes and new tutorials can be pushed to the master branch of your fork and preview before [making a pull request.](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).  To view and compare to the main documentation page, visit [`http://sciinstitute.github.io/UncertainSCI/`](http://sciinstitute.github.io/UncertainSCI/).  

### Adding Content

**This chapter provides some examples of how to add some types of content that may be needed for a tutorial.  For general Markdown info, see [here](https://www.markdownguide.org/basic-syntax/) and [here](https://guides.github.com/features/mastering-markdown/).**

**Scope: [Adding Content](#adding-content) - [Figures](#figures) - [Math](#math) - [Citations](#citations) - [Snippets](#snippets) - [Links](#links) - [Referencing Sphynx](#referencing-sphynx)**

#### Figures
Figures can be added fairly easily in Markdown, with a simple call to the location:
```
![Alt text](../assets/images/carousel_images/sample.png "Title")
```
![UncertainSCI example image](../assets/images/carousel_images/sample.png "example markdown")

However, using a bit of html allows us to <a href="#example">reference the figure</a> easier:
```
<figure id="example">
<img src="../assets/images/carousel_images/sample.png" alt="UncertainSCI example image">
<figcaption>Example for including an image in tutorial.</figcaption>
</figure>
```
And to reference:
```
<a href="#example">reference the figure</a>
```
<figure id="example">
<img src="../assets/images/carousel_images/sample.png" alt="UncertainSCI example image">
<figcaption>Example for including an image in tutorial.</figcaption>
</figure>

#### Math
Math equations can be used in Markdown using [MathJax](http://docs.mathjax.org/en/latest/basic/mathematics.html).  Mathjax will convert LaTex format:
```
$$ \frac{\partial \rho}{\partial t} + \nabla \cdot \vec{j} = 0 \,. \label{eq:continuity} $$
```
$$ \frac{\partial \rho}{\partial t} + \nabla \cdot \vec{j} = 0 \,. \label{eq:continuity} $$
It can also use MathJax specific tags:
```
\\[ x = {-b \pm \sqrt{b^2-4ac} \over 2a} \\]
```
\\[ x = {-b \pm \sqrt{b^2-4ac} \over 2a} \\]


#### Citations
Citations in Markdown uses [Pandoc](https://pandoc.org).  The citations can stored in Bibtex format

TODO.


#### Snippets

#### Links
Including links in Markdown is simple, just use `<>` or `[]()`.  For example, an internal link for section [Adding Content](#adding-content) is :
```
[Adding Content](#adding-content)
```
When using internal links to sections, include the name of the section, all lower case and with `-` replacing spaces, and all special characters ommited.  Linking to other pages in within the UncertainSCI documentation requires a relative path.  [User Documentation](../users.html) is:
```
[User Documentation](../users.html)
```
Links to other websites can include the full URL.  Using `<>` will show the URL, `[]()` will hide it with other text.  
```
<https://www.markdownguide.org>
[Markdown](https://www.markdownguide.org)
```
<https://www.markdownguide.org>
[Markdown](https://www.markdownguide.org)


#### Referencing Sphynx

TODO

To link the UncertainSCI API generated using Sphynx, Use this syntax: [:ref:`pce`](../pce.rst).  This should work but isn't yet.  

### Content Guide

TODO

### Supplemental Materials

TODO

#### Example Scripts
#### Movies
#### Datasets


