from setuptools import find_packages, setup

package_name = 'dexhand_hand_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'mediapipe',
        'numpy',
    ],

    zip_safe=True,
    maintainer='jhseon',
    maintainer_email='jhseon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_pose = dexhand_hand_pose.handpose:main'
        ],
    },
)
