.. Segwork documentation master file, created by
   sphinx-quickstart on Mon Jun 20 12:32:40 2022.
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


Content
==================

* :ref:`genindex`
* :ref:`modindex`
* `Installation`_
* `Registry`_
* `Segmentation dataset`_
* `Pixel counter and Weight algorithm`_
* `Transformations`_
.. * :ref:`search`

.. _Installation:

Installation
==================

To use Segwork, first install it using pip:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/ARubiose/segwork.git

.. _Registry:

Registry
========
.. automodule:: segwork.registry
   :members:

.. _Segmentation dataset:

Segmentation dataset
====================
.. autoclass:: segwork.data.SegmentationDataset
   :members:

.. _Pixel counter and Weight algorithm:

Pixel counter and Weight algorithm
==================================
.. automodule:: segwork.data.balance
   :members:

.. _Transformations:

Transformations
=====================
.. automodule:: segwork.data.transform
   :members:
