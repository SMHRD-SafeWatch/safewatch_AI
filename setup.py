from setuptools import setup, find_packages

setup(
    name="safewatch",
    version="0.1.0",
    description="Industrial Safety Detection System",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.5",
        "numpy>=1.21.0",
    ],
)