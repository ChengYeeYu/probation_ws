from setuptools import find_packages, setup

package_name = 'probation_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/probation.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='minh',
    maintainer_email='minhnhangia@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'solution_template = probation_bringup.solution_template:main',
            'solution = probation_bringup.solution:main',
            'minimal_publisher = probation_bringup.minimal_publisher:main',
            'minimal_subscriber = probation_bringup.minimal_subscriber:main',
            'service_member_function = probation_bringup.service_member_function:main',
            'client_member_function = probation_bringup.client_member_function:main',
        ],
    },
)
