from setuptools import setup, find_packages

setup(
    name='fluospotter',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'monai',
        'scikit-image',
        'sklearn',
        'tifffile',
        'pandas'
    ],
    author='Arnau Blanco Borrego',
    author_email='blancoarnau@gmail.com',
    description='Python library designed for nuclei segmentation and puncta detection in fluorescence microscopy images.',
    url='https://github.com/arnaublanco/Fluospotter',
)