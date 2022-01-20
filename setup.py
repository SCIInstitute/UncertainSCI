from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='UncertainSCI',
    version='0.1.1-b0',
    author='UncertainSCI Developers',
    author_email='uncertainsci@sci.utah.edu',
    packages=find_packages(),
    package_dir={'': '.'},
    download_url=r'https://github.com/SCIInstitute/UncertainSCI',
    description=r'A Non-invasive Uncertainty Quantification tool',
    long_description= long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Operating System :: Android",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
        ],
    license='MIT',
    keywords='Uncertainty Quantification, Simulation',
    url=r'https://sci.utah.edu/sci-software/simulation/uncertainsci.html',
    install_requires=[
        "wheel",
        "certifi==2020.4.5.1",
        "cycler==0.10.0",
        "kiwisolver==1.2.0",
        "matplotlib==3.1.3",
        "numpy==1.15.2; python_version < '3.8'",
        "numpy==1.17.2; python_version >= '3.8'",
        "packaging==20.3",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.1",
        "scipy==1.4.1",
        "six==1.15.0"
    ]
    )