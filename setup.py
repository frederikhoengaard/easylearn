from setuptools import setup, find_packages

with open('requirements.txt') as infile:
    requirements = infile.readlines()

setup(
    name='easy-learn',
    version = '0.1.0',
    url = 'https://github.com/frederikhoengaard/easy-learn',
    download_url = 'https://github.com/frederikhoengaard/easy-learn/tree/main/easy-learn',
    license = '',
    author = 'Frederik P. Høngaard',
    author_email = 'mail@frederikhoengaard.com',
    description = 'Machine learning model library',
    packages = find_packages(),
    install_requires = requirements
)