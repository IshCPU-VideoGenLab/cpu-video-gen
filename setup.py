from setuptools import setup, find_packages
setup(
    name="cpu-video-gen",
    version="0.1.0",
    author="Ishmael Affum Kwakye",
    description="CPU-native video generation: codec-inspired SSM architecture for commodity hardware",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IshCPU-VideoGenLab/cpu-video-gen",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0", "numpy>=1.24.0", "einops>=0.7.0", "psutil>=5.9.0"],
    entry_points={"console_scripts": ["cpu-video-gen=cpu_video_gen.cli:main"]},
)
