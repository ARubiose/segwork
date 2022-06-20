.. Segwork documentation master file, created by
   sphinx-quickstart on Sun Jun 19 18:24:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Segwork's documentation!
===================================

**Segwork** is a Python library that includes a set of tools for semantic segmentation.

.. note::

   This project is part of a Master Thesis for the European Master in Software Engineering (EMSE).

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
Installation
==================

To use Segwork, first install it using pip:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/ARubiose/segwork.git

Contents
==================

.. * :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`

ConfigurableRegistry
==================
.. automodule:: segwork.registry
   :members:
   :undoc-members:
   :show-inheritance:

SegmentationDataset
==================
.. automodule:: segwork.data.dataset
   :members:
   :undoc-members:
   :show-inheritance:

PixelCounter and WeightAlgorithm
================================
.. automodule:: segwork.data.balance
   :members:
   :undoc-members:
   :show-inheritance:

Transformation
==================
.. automodule:: segwork.data.transform
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: segwork.data.transform.generate_numpy_files


