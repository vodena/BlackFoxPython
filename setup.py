# coding: utf-8

"""
    BlackFox

    BlackFox client

    OpenAPI spec version: v1

"""


from setuptools import setup, find_packages  # noqa: H301

NAME = "blackfox"
VERSION = "0.0.8"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = ["urllib3 >= 1.15", "six >= 1.10", "certifi", "python-dateutil"]

setup(
    name=NAME,
    version=VERSION,
    description="BlackFox client",
    author="Tomislav Mrdja",
    author_email="",
    url="https://github.com/tmrdja/BlackFoxPython",
    keywords=["BlackFox"],
    install_requires=REQUIRES,
    packages=find_packages(exclude=["test.*", "test"]),
    include_package_data=True,
    long_description="""\
        BlackFox python client
    """
)
