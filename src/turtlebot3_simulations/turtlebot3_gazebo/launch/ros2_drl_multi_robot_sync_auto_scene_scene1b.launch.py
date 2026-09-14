import os
import tempfile

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _blank_to(value, fallback):
    return fallback if value is None or str(value).strip() == '' else value


def _setup_launch(context, *args, **kwargs):
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_turtlebot3_description = get_package_share_directory('turtlebot3_description')

    world_name = LaunchConfiguration('world_name').perform(context)
    world_path = os.path.join(pkg_turtlebot3_gazebo, 'worlds', world_name)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    gui = LaunchConfiguration('gui').perform(context).lower() in ('true', '1', 'yes')

    # Scene-aware default spawn points.
    # These only affect the initial Gazebo spawn. Training/eval scripts will reset poses afterwards.
    # Scene1b/bottleneck: place robots in the middle horizontal passage, near left/right walls,
    # away from the central four obstacles.
    world_name_lower = world_name.lower()
    if ('scene1b' in world_name_lower) or ('bottleneck' in world_name_lower):
        default_poses = {
            'robot1_x_pose': '-4.0',
            'robot1_y_pose': '0.0',
            'robot1_yaw': '0.0',        # face +x, toward the middle passage
            'robot2_x_pose': '4.0',
            'robot2_y_pose': '0.0',
            'robot2_yaw': '3.14159',    # face -x, toward the middle passage
        }
    elif ('scene1' in world_name_lower) or ('middle' in world_name_lower):
        default_poses = {
            'robot1_x_pose': '0.0',
            'robot1_y_pose': '-4.0',
            'robot1_yaw': '1.5708',
            'robot2_x_pose': '0.0',
            'robot2_y_pose': '4.0',
            'robot2_yaw': '-1.5708',
        }
    else:
        # Scene0/open baseline default spawn points.
        default_poses = {
            'robot1_x_pose': '-0.5',
            'robot1_y_pose': '0.6',
            'robot1_yaw': '0.0',
            'robot2_x_pose': '0.5',
            'robot2_y_pose': '-0.6',
            'robot2_yaw': '3.14159',
        }

    robot1_x = _blank_to(LaunchConfiguration('robot1_x_pose').perform(context), default_poses['robot1_x_pose'])
    robot1_y = _blank_to(LaunchConfiguration('robot1_y_pose').perform(context), default_poses['robot1_y_pose'])
    robot1_yaw = _blank_to(LaunchConfiguration('robot1_yaw').perform(context), default_poses['robot1_yaw'])
    robot2_x = _blank_to(LaunchConfiguration('robot2_x_pose').perform(context), default_poses['robot2_x_pose'])
    robot2_y = _blank_to(LaunchConfiguration('robot2_y_pose').perform(context), default_poses['robot2_y_pose'])
    robot2_yaw = _blank_to(LaunchConfiguration('robot2_yaw').perform(context), default_poses['robot2_yaw'])
    robot_z = _blank_to(LaunchConfiguration('robot_z_pose').perform(context), '0.01')

    actions = []
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_path, 'pause': 'false'}.items(),
    ))
    if gui:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
        ))
    else:
        print("[launch] gui:=false → gzclient disabled (headless eval)")

    robots = [
        {'name': 'robot1', 'x_pose': robot1_x, 'y_pose': robot1_y, 'z_pose': robot_z, 'yaw': robot1_yaw},
        {'name': 'robot2', 'x_pose': robot2_x, 'y_pose': robot2_y, 'z_pose': robot_z, 'yaw': robot2_yaw},
    ]

    for robot in robots:
        robot_name = robot['name']
        urdf_xacro_path = os.path.join(pkg_turtlebot3_description, 'urdf', 'turtlebot3_waffle.urdf.xacro')
        robot_description_config = xacro.process_file(urdf_xacro_path, mappings={'robot_namespace': robot_name})
        robot_description_content = robot_description_config.toxml()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as temp_file:
            temp_file.write(robot_description_content)
            temp_urdf_path = temp_file.name

        actions.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=robot_name,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time.lower() in ('true', '1', 'yes'),
                'robot_description': robot_description_content,
                'frame_prefix': robot_name + '/',
            }],
        ))

        actions.append(Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', temp_urdf_path,
                '-entity', robot_name,
                '-robot_namespace', robot_name,
                '-x', robot['x_pose'],
                '-y', robot['y_pose'],
                '-z', robot['z_pose'],
                '-Y', robot['yaw'],
            ],
            output='screen',
        ))

    print(f"[launch] world_name={world_name}")
    print(f"[launch] robot1=({robot1_x}, {robot1_y}, yaw={robot1_yaw}), "
          f"robot2=({robot2_x}, {robot2_y}, yaw={robot2_yaw}), z={robot_z}")
    return actions


def generate_launch_description():
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    set_gazebo_master_uri = SetEnvironmentVariable('GAZEBO_MASTER_URI', 'http://localhost:11345')
    set_model_db_off = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    local_models = os.path.join(pkg_turtlebot3_gazebo, 'models')
    combined_model_path = local_models + ((":" + existing_model_path) if existing_model_path else "")
    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', combined_model_path)

    return LaunchDescription([
        DeclareLaunchArgument('world_name', default_value='multi_robot_drl_stagec.world',
                              description='World file name under turtlebot3_gazebo/worlds'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Launch Gazebo GUI (gzclient). gui:=false for headless eval.'),

        # Empty default means: choose scene-aware default pose from world_name.
        # Passing values here still overrides the automatic scene defaults.
        DeclareLaunchArgument('robot1_x_pose', default_value=''),
        DeclareLaunchArgument('robot1_y_pose', default_value=''),
        DeclareLaunchArgument('robot1_yaw', default_value=''),
        DeclareLaunchArgument('robot2_x_pose', default_value=''),
        DeclareLaunchArgument('robot2_y_pose', default_value=''),
        DeclareLaunchArgument('robot2_yaw', default_value=''),
        DeclareLaunchArgument('robot_z_pose', default_value='0.01'),

        set_gazebo_master_uri,
        set_model_db_off,
        set_model_path,
        OpaqueFunction(function=_setup_launch),
    ])
