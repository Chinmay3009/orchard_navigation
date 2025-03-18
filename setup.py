from setuptools import setup

package_name = 'orch_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # Include the package name here
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch_urdf_into_gazebo.launch.py']),
        ('share/' + package_name + '/config', ['config/nav2_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Row following node for orchard simulation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'row_following_node = orch_sim.navigation:main',  # Ensure the main() function exists in your script
        ],
    },
)

