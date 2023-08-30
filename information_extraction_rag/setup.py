from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README.md file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line]

extra_requirements = {}

with open(path.join(here, "requirements_test.txt"), encoding="utf-8") as f:
    extra_requirements["test"] = [line.strip() for line in f if line]

setup(
    name="info-extract",
    version="0.0.1",
    description="Extract structured data from unstructured documents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wael Abid, Geoffrey Angus, Jeffery Kinnison",
    keywords="llm information_extraction",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extra_requirements,
)
