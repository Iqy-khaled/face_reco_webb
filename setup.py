from setuptools import setup, find_packages

setup(
    name="face-recognition-attendance",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.0.1",
        "numpy>=1.21.1",
        "tensorflow>=2.15.0",
        "pillow>=8.3.1",
        "python-dotenv>=0.19.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Face Recognition Attendance System",
    keywords="face, recognition, attendance, computer-vision",
    python_requires=">=3.7",
)
