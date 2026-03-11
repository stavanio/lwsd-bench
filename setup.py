from setuptools import find_packages, setup
import os
from glob import glob

package_name = "lwsd_bench"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
    ],
    zip_safe=True,
    maintainer="Stavan Dholakia",
    maintainer_email="stavan@mefferdi.com",
    description="LWSD diagnostic for foundation model robots",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lwsd_monitor = lwsd_bench.lwsd_monitor:main",
            "lwsd_bag_analyzer = lwsd_bench.bag_analyzer:main",
            "latency_injector = lwsd_bench.latency_injector:main",
            "ground_truth_publisher = lwsd_bench.ground_truth_publisher:main",
        ],
    },
)
