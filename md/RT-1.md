# RT-1

```shell
pytorch_rt1_with_trainer_and_tester-11148a/
├── IO_trainer_torch_space_change.py           "Code for training RT-1"
├── train_config.json            "Configuration file for training RT-1"
├── cube_test_all_400.py                  "RT-1 test code for the cube"
├── apple_test_all_400.py                "RT-1 test code for the apple"
├── tea_box_test_all_400.py            "RT-1 test code for the tea box"
```

## Environment preparation

1. Create a virtual environment

2. Install dependencies

```shell
cd pytorch_rt1_with_trainer_and_tester-11148a
pip install -r requirements.txt
```

## Data preparation

Generate a dataset with the following structure through [Simulator Data Collection](../md/simulator_data_collection.md), or use the dataset we provide. The link to our dataset is as follows:[https://www.alipan.com/t/A5OQ8SzhMnHG44K3Q7uX](https://www.alipan.com/t/A5OQ8SzhMnHG44K3Q7uX)

```shell
cube/
├── front/
│   ├── data_000/
│   │   ├── result.csv
│   │   ├── result_raw.csv
│   │   └── rgb/
│   ├── data_001/
│   ├── ...
│   └── dataset_info.json
├── root/
├── side/
├── topdown/
└── wrist/
```

## Train RT-1

### Configure the `train_config.json` file

- **data_path**: The path to the dataset that contains the data used for training.

- **cam_view**: Defines the camera views being used, here set to `["front", "wrist"]`, meaning data from the front and wrist perspectives will be utilized.

- **log_dir**: The directory for saving training logs, used to record information and metrics during the training process.

- **batch_size**: The number of samples used in each training iteration, set to `8`.

- **epochs**: The total number of training epochs, set to `150`, meaning the model will be trained for 150 cycles.

- **val_interval**: The interval for validation, set to `2`, meaning validation will occur every 2 training epochs.

### Run IO_trainer_torch_space_change.py

```python
python IO_trainer_torch_space_change.py
```

## Test RT-1

We take `cube_test_all_400.py` as an example.

1、Change `data_path` to the path of your own dataset and set `cam_view` to the camera perspectives used during RT-1 training.

```python
args = {
    "mode": "train",
    "device": "cuda:1",
    "data_path": "/data/yuantingyu/cube/",
    "cam_view": ["front", "wrist","side"],
```

2、Change the checkpoint path.

```python
args["resume_from_checkpoint"] = ""
```

3、Set the URDF file for the object to be grasped, along with the scale factor, initial position, and initial orientation.

```python
self.tar_obj = p.loadURDF("urdf/cube/cube.urdf", [0, 0, 0], globalScaling=0.04)
```

```python
        pos = [-0.28, 0.48, 0.645]
        rot = p.getQuaternionFromEuler([0, np.pi / 2, r])
```

4、Set the data save path.

```
video_dir = ""
excel_path = ""
```

5、Run

```shell
python cube_test_all_400.py
```

## Demonstration video of testing the RT-1 model

https://github.com/user-attachments/assets/fe95aa74-a1fb-47d9-ac51-6b9bba1cd2d8
