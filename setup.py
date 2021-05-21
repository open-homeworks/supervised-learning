  
from setuptools import setup, find_packages

required = ["numpy"]
dev_required = ["pytest", "pytest-xdist", "pytest-cov", "black", "mypy", "pydocstyle"]

setup(
    name="supervised_learning",
    version="0.0.1",
    description="A companion to study machine learning",
    url="https://github.com/open-workbooks/supervised-learning-workbook",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=required,
    include_package_data=True,
    extras_require={"dev": dev_required},
    package_dir={"": "."},
)