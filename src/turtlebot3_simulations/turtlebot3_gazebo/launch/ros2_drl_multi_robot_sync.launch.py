import os
import tempfile

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    set_gazebo_master_uri = SetEnvironmentVariable('GAZEBO_MASTER_URI', 'http://localhost:11345')

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_turtlebot3_description = get_package_share_directory('turtlebot3_description')

    set_model_db_off = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    local_models = os.path.join(pkg_turtlebot3_gazebo, 'models')
    combined_model_path = local_models + ((":" + existing_model_path) if existing_model_path else "")
    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', combined_model_path)

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='multi_robot_drl_stagec.world',
        description='World file name under turtlebot3_gazebo/worlds'
    )

    world = PathJoinSubstitution([
        pkg_turtlebot3_gazebo,
        'worlds',
        LaunchConfiguration('world_name')
    ])

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world, 'pause': 'false'}.items(),
    )
    # GUI (gzclient) can be disabled for headless eval runs: gui:=false
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Whether to launch the Gazebo GUI (gzclient). Set gui:=false for headless eval.'
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    robots = [
        {'name': 'robot1', 'x_pose': '-3.0', 'y_pose': '-3.0', 'z_pose': '0.01'},
        {'name': 'robot2', 'x_pose': '3.0', 'y_pose': '3.0', 'z_pose': '0.01'},
    ]

    ld = LaunchDescription()
    ld.add_action(world_name_arg)
    ld.add_action(gui_arg)
    ld.add_action(set_gazebo_master_uri)
    ld.add_action(set_model_db_off)
    ld.add_action(set_model_path)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    for robot in robots:
        robot_name = robot['name']
        urdf_xacro_path = os.path.join(pkg_turtlebot3_description, 'urdf', 'turtlebot3_waffle.urdf.xacro')
        robot_description_config = xacro.process_file(urdf_xacro_path, mappings={'robot_namespace': robot_name})
        robot_description_content = robot_description_config.toxml()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as temp_file:
            temp_file.write(robot_description_content)
            temp_urdf_path = temp_file.name

        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=robot_name,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description_content,
                'frame_prefix': robot_name + '/',
            }],
        )

        spawn_entity = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', temp_urdf_path,
                '-entity', robot_name,
                '-robot_namespace', robot_name,
                '-x', robot['x_pose'],
                '-y', robot['y_pose'],
                '-z', robot['z_pose'],
            ],
            output='screen',
        )

        ld.add_action(robot_state_publisher)
        ld.add_action(spawn_entity)

    return ld
