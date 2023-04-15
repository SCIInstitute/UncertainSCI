from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='UncertainSCI',
    version='1.0rc2',
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
        "matplotlib>=3.1.3",
        "numpy>=1.22",
        "scipy>=1.4.1"
    ]
    )
