import os
import xacro
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction, RegisterEventHandler, ExecuteProcess, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Set GAZEBO_MASTER_URI to an empty string to force offline startup
    set_gazebo_master_uri = SetEnvironmentVariable(
        'GAZEBO_MASTER_URI',
        'http://localhost:11345'  # Use a local URI
    )

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_turtlebot3_description = get_package_share_directory('turtlebot3_description')

    # Ensure Gazebo uses local models only (no online fetch)
    set_model_db_off = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')
    # Ensure local model paths include this package's models
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    local_models = os.path.join(pkg_turtlebot3_gazebo, 'models')
    combined_model_path = (local_models + (":" + existing_model_path if existing_model_path else ""))
    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', combined_model_path)

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # World file
    world_file_name = 'multi_robot_drl.world'
    world = os.path.join(pkg_turtlebot3_gazebo, 'worlds', world_file_name)

    # Gazebo launch
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world, 'pause': 'false'}.items() # Start Gazebo unpaused for immediate sensor data
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # List of robots to spawn
    robots = [
        {'name': 'robot1', 'x_pose': '0.0', 'y_pose': '0.5', 'z_pose': '0.01'},
    ]

    spawn_robots_cmds = []
    last_spawn_action = None  # Track the last spawn_entity.py node
    
    for robot in robots:
        robot_name = robot['name']
        
        # Process the URDF file with xacro
        urdf_xacro_path = os.path.join(pkg_turtlebot3_description, 'urdf', 'turtlebot3_waffle.urdf.xacro')
        robot_description_config = xacro.process_file(
            urdf_xacro_path,
            mappings={'robot_namespace': robot_name}
        )
        robot_description_content = robot_description_config.toxml()

        # Create a temporary file to store the robot description
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as temp_file:
            temp_file.write(robot_description_content)
            temp_urdf_path = temp_file.name

        # Robot State Publisher
        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=robot_name,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description_content,
                'frame_prefix': robot_name + '/'
            }],
        )

        # Spawn Robot from the temporary file
        spawn_entity = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', temp_urdf_path,
                '-entity', robot_name,
                '-robot_namespace', robot_name,
                '-x', robot['x_pose'],
                '-y', robot['y_pose'],
                '-z', robot['z_pose']
            ],
            output='screen',
        )
        
        # Keep track of the last spawn action
        last_spawn_action = spawn_entity
        
        spawn_robots_cmds.append(
            GroupAction([
                robot_state_publisher,
                spawn_entity,
            ])
        )

    # No unpause handler needed since we start unpaused

    # Create the launch description and populate
    ld = LaunchDescription()

    # Apply environment settings early
    ld.add_action(set_gazebo_master_uri)
    ld.add_action(set_model_db_off)
    ld.add_action(set_model_path)

    # Add the actions to launch all of the nodes
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    for cmd in spawn_robots_cmds:
        ld.add_action(cmd)
    

    return ld
