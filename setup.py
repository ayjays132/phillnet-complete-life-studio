from setuptools import setup, find_packages

setup(
    name="life_game_env",
    version="0.0.1",
    author="Manus AI",
    description="A Gym environment for a Life Simulation game powered by the Phillnet-CompleteLife dataset.",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "datasets",
        "numpy",
    ],
    python_requires=">=3.8",
)
