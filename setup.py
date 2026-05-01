from setuptools import find_packages, setup

setup(
    name="mem_update_bench",
    version="0.1.0",
    description="MemUpdateBench: repeated same-slot update-frequency benchmark for external memory systems",
    packages=find_packages(),
    python_requires=">=3.10",
)
