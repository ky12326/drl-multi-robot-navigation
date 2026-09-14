import os
import tempfile

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Per-robot pose args are declared for robots 1..MAX_ROBOTS; beyond that a
# geometric fallback pose is used (the env teleports robots on reset anyway).
MAX_ROBOTS = 6


def _blank_to(value, fallback):
    return fallback if value is None or str(value).strip() == '' else value


def _scene_default_poses(world_name):
    """Scene-aware default spawn poses, one (x, y) per robot (1-indexed)."""
    if ('scene2' in world_name) or ('double_bottleneck' in world_name):
        return [('-4.0', '0.95'), ('4.0', '-0.95'), ('0.0', '-3.0'),
                ('-3.0', '2.5'), ('3.0', '2.5'), ('0.0', '3.0')]
    if ('scene1' in world_name) or ('middle' in world_name):
        return [('-3.5', '-3.5'), ('3.5', '3.5'), ('0.0', '0.0'),
                ('-3.5', '2.5'), ('3.5', '-2.5'), ('0.0', '-2.0')]
    return [('-0.5', '0.6'), ('0.5', '-0.6'), ('0.8', '0.8'),
            ('-0.8', '-0.8'), ('0.0', '1.2'), ('1.5', '0.0')]


def _default_pose(index, world_name, num_robots):
    """Scene-aware default spawn pose for robot `index` (1-indexed)."""
    table = _scene_default_poses(world_name)
    if index <= len(table):
        return table[index - 1]
    # Geometric fallback: spread extra robots along the bottom edge.
    span = 6.0
    denom = max(num_robots - 1, 1)
    x = -3.0 + ((index - 1) % max(num_robots, 2)) * (span / denom)
    return (f'{x:.2f}', '-3.2')


def _setup_launch(context, *args, **kwargs):
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_turtlebot3_description = get_package_share_directory('turtlebot3_description')

    world_name = LaunchConfiguration('world_name').perform(context)
    world_path = os.path.join(pkg_turtlebot3_gazebo, 'worlds', world_name)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    gui = LaunchConfiguration('gui').perform(context).lower() in ('true', '1', 'yes')
    num_robots = int(LaunchConfiguration('num_robots').perform(context) or '3')
    robot_z = _blank_to(LaunchConfiguration('robot_z_pose').perform(context), '0.01')

    def _pose_arg(name, fallback):
        try:
            return _blank_to(LaunchConfiguration(name).perform(context), fallback)
        except Exception:
            return fallback

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

    robots = []
    for i in range(1, num_robots + 1):
        x_default, y_default = _default_pose(i, world_name, num_robots)
        robots.append({
            'name': f'robot{i}',
            'x_pose': _pose_arg(f'robot{i}_x_pose', x_default),
            'y_pose': _pose_arg(f'robot{i}_y_pose', y_default),
            'z_pose': robot_z,
        })

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
            ],
            output='screen',
        ))

    print(f"[launch] world_name={world_name}, num_robots={num_robots}")
    for r in robots:
        print(f"[launch] {r['name']}=({r['x_pose']}, {r['y_pose']})")
    return actions


def generate_launch_description():
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    set_gazebo_master_uri = SetEnvironmentVariable('GAZEBO_MASTER_URI', 'http://localhost:11345')
    set_model_db_off = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    local_models = os.path.join(pkg_turtlebot3_gazebo, 'models')
    combined_model_path = local_models + ((":" + existing_model_path) if existing_model_path else "")
    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', combined_model_path)

    args = [
        DeclareLaunchArgument('world_name', default_value='multi_robot_drl_scene2_double_bottleneck.world',
                              description='World file name under turtlebot3_gazebo/worlds'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Launch Gazebo GUI (gzclient). gui:=false for headless eval.'),
        DeclareLaunchArgument('num_robots', default_value='3',
                              description='Number of robots to spawn'),
    ]
    for i in range(1, MAX_ROBOTS + 1):
        args.append(DeclareLaunchArgument(f'robot{i}_x_pose', default_value=''))
        args.append(DeclareLaunchArgument(f'robot{i}_y_pose', default_value=''))
    args.append(DeclareLaunchArgument('robot_z_pose', default_value='0.01'))
    args += [set_gazebo_master_uri, set_model_db_off, set_model_path,
             OpaqueFunction(function=_setup_launch)]
    return LaunchDescription(args)
