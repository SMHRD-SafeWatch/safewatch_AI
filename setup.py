from setuptools import setup, find_packages

setup(
    name="safewatch",
    version="0.1.0",
    description="Industrial Safety Detection System",
    packages=find_packages(),
    python_requires=">=3.8",  
    install_requires=[
        "fastapi>=0.109.2",
        "uvicorn>=0.27.1",
        "opencv-python>=4.9.0.80",
        "ultralytics>=8.1.27",
        "pydantic>=2.6.1",
        "python-multipart>=0.0.6",
        "numpy>=1.26.4",
        "python-dotenv>=1.0.1",
        "cx-Oracle>=8.3.0"
    ],
)