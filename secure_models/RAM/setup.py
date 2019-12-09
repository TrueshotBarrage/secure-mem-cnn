import os
import sys
import platform
from setuptools import setup, find_packages
from codecs import open

here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join("src", "pyoram", "__about__.py")) as f:
    exec(f.read(), about)

# Get the long description from the README file
def _readme():
    with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
        return f.read()

setup_requirements = []
requirements = ['cryptography',
                'paramiko',
                'boto3',
                'six',
                'tqdm']

if platform.python_implementation() == "PyPy":
    if sys.pypy_version_info < (2, 6):
        raise RuntimeError(
            "PyORAM is not compatible with PyPy < 2.6. Please "
            "upgrade PyPy to use this library.")
else:
    if sys.version_info <= (2, 6):
        raise RuntimeError(
            "PyORAM is not compatible with Python < 2.7. Please "
            "upgrade Python to use this library.")
    requirements.append("cffi>=1.0.0")
    setup_requirements.append("cffi>=1.0.0")

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__summary__'],
    long_description=_readme(),
    url=about['__uri__'],
    author=about['__author__'],
    author_email=about['__email__'],
    license=about['__license__'],
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Security :: Cryptography',
        "Natural Language :: English",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    keywords='oram, storage, privacy, cryptography, cloud storage',
    package_dir={'': 'src'},
    packages=find_packages(where="src", exclude=["_cffi_src", "_cffi_src.*"]),
    setup_requires=setup_requirements,
    install_requires=requirements,
    cffi_modules=["src/_cffi_src/virtual_heap_helper_build.py:ffi"],
    # use MANIFEST.in
    include_package_data=True,
    test_suite='nose2.collector.collector',
    tests_require=['unittest','nose2']
)
