# Test-Yolov3-by-Hoang-Vu
# 1. Install requirement library in requirement.txt
```bash
pip install -r requirements.txt
```

# 2. Dataset
Both annotation and image file in 1 folder

Folder img_hand contain 2 folders "train" and "test" 

Access file XML_to_YOLOv3.py in folder tools

Change relative directory: 

* data_dir = '/hand_img/' # path of folder contain train and test
       
* Dataset_name_path = "model_data/hand_names.txt"
       
* Dataset_train = "model_data/hand_train.txt"
       
* Dataset_test = "model_data/hand_test.txt"
  
Run file XML_to_YOLOv3.py
  ```python
  python XML_to_YOLOv3.py
  ```
  
# 3. Running
In file configs.py of folder yolov3 change: 
* YOLO_TYPE
* TRAIN_ANNOT_PATH = "./model_data/hand_train.txt"
* TEST_ANNOT_PATH = "./model_data/hand_test.txt"
* TRAIN_CLASSES = "./model_data/hand_names.txt"

```python
python train.py
```
