# saris


.. image:: https://img.shields.io/pypi/v/saris.svg
        :target: https://pypi.python.org/pypi/saris

.. image:: https://img.shields.io/travis/hieutrungle/saris.svg
        :target: https://travis-ci.com/hieutrungle/saris

.. image:: https://readthedocs.org/projects/saris/badge/?version=latest
        :target: https://saris.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Self Adjustable Reconfigurable Intelligent Surfaces (SARIS)


* Free software: MIT
* Documentation: https://saris.readthedocs.io.

## Instructions

## Docker

```bash
docker build -t pytorch-saris . -f Dockerfile
docker run --rm --runtime=nvidia --gpus all -it saris
```

# Installation

Use Python 3.10

```bash
cd /path/to/saris
pip install -e .
pip install torch==2.5.1
pip install -r requirements.txt
```

## Features

* TODO

## Credits

This package was created with Cookiecutter_ and the `briggySmalls/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`briggySmalls/cookiecutter-pypackage`: https://github.com/briggySmalls/cookiecutter-pypackage
