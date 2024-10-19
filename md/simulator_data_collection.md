# Simulator Data Collection

## Environment preparation

1. Create a virtual environment

2. Install dependencies

```shell
cd pytorch_rt1_with_trainer_and_tester-11148a
pip install -r requirements.txt
```

Since data collection in the simulation environment involves calling GraspNet and object detection, GraspNet and object detection APIs are also required in the environment setup. The API files can be downloaded from here:[https://www.alipan.com/t/4DDctQIrZrPwwALMvLVK](https://www.alipan.com/t/4DDctQIrZrPwwALMvLVK)

After downloading the API files, it is necessary to set the correct paths in the related files.

```python
parser.add_argument('--checkpoint_path', default="/home/taizun/embodiedAI/graspnet-baseline/checkpoint-rs.tar",help='Model checkpoint path')
data_dir = '/home/taizun/embodiedAI/graspnet-baseline/doc/mydata'
```

## File configuration

### Related files

The following Python files correspond to programs that collect grasping data for different objects in the virtual environment. 

- fangzhen_delta_change_cube.py

- fangzhen_urdf_change_apple.py

- fangzhen_urdf_change_tea_box.py

These Python files will record videos while collecting data.

- fangzhen_urdf_change_apple_vedio.py

- fangzhen_urdf_changecube_vedio.py

- fangzhen_urdf_changetea_box_vedio.py

### Configuration

File configuration using the file `fangzhen_urdf_change_cube_vedio.py` as an example.

1、Set the camera perspective for data collection.

```python
CAM_INFO = {
    "front": [[0, 0, 0.7], 1.8, 180, -20, 0, 40],
    "fronttop": [[0, 0.5, 0.7], 1.5, 180, -60, 0, 35],
    "topdown": [[0, 0.35, 0], 2.0, 0, -90, 0, 45],
    "side": [[0, 0.35, 0.9], 1.5, 90, 0, 0, 40],
    "root": [[0, 0.6, 0.75], 1.3, -35, -5, 0, 40],
    "wrist": [],
}: [],
}
```

2、Initialize the environment and load the URDF file. (The URDF file can be customized.)

```python
    def load_env(self):
        p.loadURDF("table/table.urdf", [0, 0.35, 0], [0, 0, 0, 1])
        self.tar_obj = p.loadURDF("urdf/plastic_apple/model.urdf", [0, 0, 0], globalScaling=0.75)
```

3、Set the initial position and orientation of the grasping object.

```python
        pos = [x, y, 0.645]
        rot = p.getQuaternionFromEuler([0, np.pi / 2, r])
```

## Start data collection

```shell
python fangzhen_urdf_change_cube_vedio.py
```

https://github.com/user-attachments/assets/4f7b001a-2042-4a98-9e46-6b51b9a78c99
