import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data_pipes_mat",
    version="0.0.1",
    author="Arnaud tauveron",
    author_email="atauveron.pro@gmail.com",
    description="A small example package with all practices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "competition_project\data_science_test_AT"},
    packages=setuptools.find_packages(where="competition_project\data_science_test_AT"),
    python_requires=">=3.6",
)
