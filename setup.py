from setuptools import setup, find_packages

setup(
    name='fluospotter',
    version='0.2',
    packages=find_packages(),
    license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    install_requires=[
        'numpy',
        'torch',
        'monai',
        'scikit-image',
        'scikit-learn',
        'tqdm',
        'tifffile',
        'pandas'
    ],
    author='Arnau Blanco Borrego',
    author_email='blancoarnau@gmail.com',
    description='Python library designed for nuclei segmentation and puncta detection in fluorescence microscopy images.',
    url='https://github.com/arnaublanco/Fluospotter',
)