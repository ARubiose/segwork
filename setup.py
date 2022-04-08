from setuptools import setup,find_packages

setup(
   name='segwork',
   version='0.1.dev',
   description='A useful framework for semantic segmentation',
   author='Alvaro Rubio',
   author_email='alvaro.rubio.segovia"gmail.com',
   url='https://github.com/pypa/sampleproject',
   keywords="deep learning, semantic segmentation, development", 
   packages=find_packages(include=['segwork', 'segwork.*']),
   install_requires=['timm'], #external packages as dependencies
)