# pku baidu kaggle dataset api

## Additional data
- car model correspondence is obtained from [ApolloScape dataset](# https://raw.githubusercontent.com/ApolloScapeAuto/dataset-api/master/car_instance/car_models.py)
- car model pickle files hosted on [Kaggle](https://www.kaggle.com/c/pku-autonomous-driving/data) have issues. The older version of the pickle files can be found at [this issue](https://github.com/ApolloScapeAuto/dataset-api/issues/1). It is saved to `data` folder. The corresponding json files are also added to `data` folder to supplement the ApolloScape dataset. 
- Out of the test images some are flipped. These are annotated and saved to [data folder](data).

## average vehicle size
```python
{'2x': {'W': 1.81794264,
  'H': 1.47786305,
  'L': 4.49547776,
  'model': 'bieke-yinglang-XT'},
 'SUV': {'W': 2.10604523,
  'H': 1.67994469,
  'L': 4.73350861,
  'model': 'biyadi-tang'},
 '3x': {'W': 1.9739563700000002,
  'H': 1.4896684399999998,
  'L': 4.83009344,
  'model': 'dazhongmaiteng'}}  
```

![](assets/vehicle_size.png)

## image and mask renderer
Run the command to render image, and get amodal bboxes.
```
python car_renderer.py
```

Original image
![](assets/ID_001d6829a.png)

Rendered mask and bbox
![](assets/ID_001d6829a_render.png)