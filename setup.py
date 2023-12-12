from setuptools import setup, find_packages 
  
setup( 
    name='k_dedede', 
    version='0.0.0', 
    description='A sample Python package', 
    author='Zony Yu', 
    author_email='zony249@gmail.com', 
    packages=['k_dedede', 'k_dedede.glue', 'k_dedede.glue.src', 'k_dedede.fdistill'], 
    install_requires=[  
        "transformers", 
        "datasets",
        "spacy>=3", 
        "allennlp==2.10.1", 
        "nltk",
    ], 
) 