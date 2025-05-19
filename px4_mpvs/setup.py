from setuptools import setup, find_packages
from glob import glob
import os
package_name = 'px4_mpvs'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('px4_mpvs/launch', '*launch.[pxy][yma]*'))),
        # (os.path.join('share', package_name), glob('launch/*.[pxy][yma]*')),
        (os.path.join('share', package_name), glob(os.path.join('px4_mpvs/config', '*.rviz'))),
        (os.path.join('share', package_name), glob(os.path.join('px4_mpvs/utils', '*.py'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tafarrel',
    maintainer_email='kupuk23@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpvs_spacecraft = px4_mpvs.mpvs_spacecraft:main',
            'test_missalignment = px4_mpvs.test.test_missalignment:main',
            'test_pose_camera = px4_mpvs.test.test_pose_camera:main',
            'ibvs_main = px4_mpvs.ibvs_main:main',
        ],
    },
)
