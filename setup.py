import setuptools

setuptools.setup(
    name="CloudyGalaxyInference",
    version="0.1",
    author="Dirk Scholte",
    author_email="dirk.scholte.20@ucl.ac.uk",
    description="Python package to fit photoionization models to emission line measurements using SBI.",
    packages=["CloudyGalaxyInference"],
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "pandas",
        "corner",
        "sbi",
        "torch",
        "dust_extinction",
        "dustmaps",
    ],
)
