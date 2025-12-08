Use this command to change the camera view:

```gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 2000 --req "pose: {position: {x: 0.0, y: 0.0, z: 15.0} orientation: {x: 0.0, y: 0.7071, z: 0.0, w: 0.7071}}"```


The recommended template folder from official tutorial has 4 packages. Notice that these packages have `ament_cmake`and different structures compared to Python packages defined by `ament_python`.

To place a new script under `_application`, we need to put it under `/scripts` folder, modify `package.xml`, `CMakeLists.txt`, and add `LaunchDescription` in the `.launch.py` file

+ `ros_gz_example_application`: holds ROS 2 specific code and configurations. Namely where control, planning or any high level algoritms reside.

+ `ros_gz_example_bringup`: holds launch files and high level utilities, communication bridge between ROS and Gazebo. Any robot or hardware specific configurations go here. If we want to load a configruation file to launch the bridge node, then we should place our configuration file right inside the folder.

+ `ros_gz_example_description`: holds the SDF description of the simulated system and any other simulation assets.

+ `ros_gz_example_gazebo` :holds Gazebo specific code and configurations. Namely this is where user-defined worlds and custom system plugins end up.


```ros2 launch ros_gz_example_bringup diff_drive.launch.py```
