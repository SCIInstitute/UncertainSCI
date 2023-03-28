# Contribution Guide

Thank you for you contributions to UncertainSCI!  We welcome and appreciate and contributions, from reporting a bugs to code contributions.  If you wish to contribute, please do so in the following ways.

## Community Support 

A great way to start contributing to UncertainSCI is to submit and answer questions on our [discussion board]<https://github.com/SCIInstitute/UncertainSCI/discussions>.   

Other ways of contacting the communtity are located on our [support page](../user_docs/support.html#support)

## Bugs and Features

We  encourage users to report any bugs they find and request any features they'd like as a [GitHub issue]<https://github.com/SCIInstitute/UncertainSCI/issues>.  If you would like to tackle any issues, please volunteer by commenting in the issue or [assigning yourself]<https://docs.github.com/en/issues/tracking-your-work-with-issues/assigning-issues-and-pull-requests-to-other-github-users>.  


## Make a Tutorial

If you have a tutorial you'd like to share, we'd love to have it.  We have a [Tutorial Tutorial](HowToTutorial.html) to explain how to make and contribute tutorials.


## Contribute Code

We appreciate to code maintenance and development that our community can provide.  If you'd like to submit a bug fix, dependency update, or an added feature, please keep in mind the [style guide](#style-guide), create a [fork](#fork-repo) of the UncertainSCI repo, and use a [Pull Request](#pull-requests) to add it to UncertainSCI.  

It is best practice to make sure that there is  [GitHub issue]<https://github.com/SCIInstitute/UncertainSCI/issues> to describe the required changes to the code, and having these issues documented become more important with the scope of the additions and changes.  Possible additions can also be discussed on our [discussion board]<https://github.com/SCIInstitute/UncertainSCI/discussions>.  

### Fork Repo

With your own github account, go to the [UncertainSCI Github page](https://github.com/SCIInstitute/UncertainSCI). Click the fork button on the upper right side of the page. It will ask you where to move the fork to, chose your own account. Once the repository is forked, clone it to your local machine with the following command.  

```
$git clone https://github.com/[yourgithubaccount]/UncertainSCI.git
```

After the the code is cloned, navigate to the repository directory and add the upstream path to the original UncertainSCI repository.  

```
$git remote add upstream https://github.com/SCIInstitute/UncertainSCI.git
```

You should be able to see both your and the original repository when you use the command:     

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

### Pull Requests

With the contributions added to a branch on a fork of UncertainSCI, it is ready to create a [pull request]<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>.  While this can be done in many ways, the easiest is probably through the web page of the forked repo. When navigating to the main page, it will usually display the a `contribute` button near the top of the page for recently updated branches.  This is a shortcut to creating a pull request to the main repo and branch.  Alternatively, a pull request can be created from the pull request tab of either the main repo or the fork.  Before making a pull request, please make sure that you've tried your best to follow the [style guide](#style-guide), and that the branch is up-to-date with the lastest master branch.  Also, please update or add [testing](#testing) as appropriate.  

Once the pull request is created, the maintainers of UncertainSCI will assign reviewers who will test and review the code to ensure that it meets the requirements of the style guide and is stable.  It is best to limit the size of each pull request to facilitate review, so if there are major new additions, please add a [GitHub issue]<https://github.com/SCIInstitute/UncertainSCI/issues> to track the progress.  


## Syle Guide

If you are editing code, take a few minutes to look at the code around you and determine its style.  Please try to keep the style of new code as similar as possible to unchanged code to avoid jarring inconsistencies in the style.

### Python 

- Indentation is 4 spaces per level for consistency with the rest of the code base. This may be revisited in the future. Do not use tabs.
- Text encoding: UTF-8 is preferred, Latin-1 is acceptable
- Comparisons:
    - To singletons (e.g. None): use ‘is’ or ‘is not’, never equality operations.
    - To Booleans (True, False): don’t ever compare with True or False (for further explanation, see PEP 8).
- Prefix class definitions with two blank lines
- Imports
    - Grouped in order of scope/commonallity
        - Standard library imports
        - Related third party imports
        - Local apps/library specific imports
            - UncertainSCI application imports and local/module imports may be grouped independently.
    - One package per line (with or without multiple function/module/class imports from the package)
- Avoid extraneous whitespaces

### Demos and Usecases

New demos and usecases are always welcome.  Please add self-contained scripts demonstrating core functionality to the [demos folder]<https://github.com/SCIInstitute/UncertainSCI/tree/master/demos>.   Demos that require external packages can be located in seperate repos, such as this [UQ BEM heart position usecase]<https://github.com/SCIInstitute/UQExampleBEMHeartPosition>

### Testing

In addition to demos, please add unit testing to new function contributed to UncertainSCI using pytest.  Unit test should be placed in the [test folder]<https://github.com/SCIInstitute/UncertainSCI/tree/master/tests>, which contains several tests to use as examples.  To run the test, use the command: 

```
pytest tests
```
 



