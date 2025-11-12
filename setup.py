from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fruit-classifier-app",
    version="1.0.0",
    author="Fruit Classification Team",
    author_email="example@example.com",
    description="A production-ready fruit classification application using Vision Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fruit-classifier-app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fruit-classifier=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["class_names.json", "README.md", "requirements.txt"],
    },
)