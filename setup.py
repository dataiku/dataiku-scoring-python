#!/usr/bin/env python

import setuptools

long_description = (open('README.md').read() + '\n\n' +
                    open('HISTORY.txt').read())

VERSION = "13.5.3"

setuptools.setup(
    name='dataiku-scoring',
    version=VERSION,
    license="Apache Software License",
    packages=setuptools.find_packages(),
    description="Dataiku ML Scoring Python library",
    long_description=long_description,
    author="Dataiku",
    author_email="support@dataiku.com",
    url="https://www.dataiku.com",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        "numpy"
    ]
)
