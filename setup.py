from setuptools import setup,find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
   name='segwork',
   version='0.1',
   description='A useful set of tools for semantic segmentation',
   long_description=long_description,
   author='Alvaro Rubio',
   author_email='alvaro.rubio.segovia"gmail.com',
   url='https://github.com/ARubiose/segwork',
   keywords="deep learning, semantic segmentation, development", 
   packages=find_packages(include=['segwork', 'segwork.*']),
   install_requires=['torch', 'torchvision'], #external packages as dependencies
)