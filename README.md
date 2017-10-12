# sdcnd

### Tips and tricks
* Grow GPU memory as required by the program (otherwise cudnn errors!) 
[Link to guthub issue](https://github.com/tensorflow/tensorflow/issues/6698)
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```
