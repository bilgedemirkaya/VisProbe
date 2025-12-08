from setuptools import setup, find_packages

setup(
    name="visprobe",
    version="0.1.0",
    description="Interactive Robustness Testing for Computer Vision Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bilge Demirkaya",
    author_email="bilgedemirkaya07@gmail.com",
    url="https://github.com/bilgedemirkaya/VisProbe",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Topic :: Security",
    ],
    keywords="adversarial-robustness computer-vision deep-learning neural-networks pytorch testing fuzzing security",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.8",

    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy<2.0.0",
        "adversarial-robustness-toolbox>=1.18.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "pandas>=2.0.0",
        "pillow>=8.0.0",  # For image processing
    ],

    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0"],
        "viz": ["altair>=4.2.0"],
        "ai-docs": ["anthropic>=0.18.0"],
    },

    entry_points={
        'console_scripts': [
            'visprobe=visprobe.cli.cli:main',
            'visprobe-update-docs=scripts.update_docs:main',
            'visprobe-auto-docs=scripts.auto_update_docs:main',
        ],
    },
)