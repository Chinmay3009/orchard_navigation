from setuptools import setup

package_name = 'row_following_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],  # Make sure the package name is correct
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A ROS 2 Python package for row following',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'row_following_node = row_following_pkg.row_following_node:main',
        ],
    },
)

