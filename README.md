# SegWork

SegWork is a Python library that includes a set of tools for semantic segmentation.

The main feature of this library are:
*   Configurabel registry: A flexible registry of classes that includes a factory method for components.
*   Segmentation dataset: An abstraction of a segmentation dataset for loading images and segmentation masks. It includes PixelCounter and WeightAlgorithm abstract classes to balance datasets.
*   Transformation: Two transformation, one to convert color masks to index masks and other to perform the opposite operation based on a color map.

## Project documentation

## Installation
```bash
pip install git+https://github.com/ARubiose/segwork.git
```

## Authors
*   **Álvaro Rubio** - [Arubiose](https://github.com/ARubiose)

## License
TODO
