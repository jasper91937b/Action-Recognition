# Install Dependency
for python >= 3.6 and tensorflow2.x

openpose :
```bash
$ conda create --name tf python=3.8
$ conda activate tf
$ git clone https://github.com/jasper91937b/Action-Recognition.git
$ conda install tensorflow  

$ cd Action_Recognition/myapp/Realtime-Action-Recognition-master/src/githubs/githubs   
$ git clone https://github.com/gsethi2409/tf-pose-estimation.git
$ cd tf-pose-estimation
$ pip install -r requirements.txt

$ conda install swig
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
$ pip install opencv-python
$ pip install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib

$ cd ../..
$ cd models/graph/cmu
$ bash download.sh
$ cd ../../..

# test openpose
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```

```bash
$ pip install scikit-learn==0.22.2.post1
$ pip install pyyaml
$ pip install simplejson
$ pip install django
$ cd Action_Recognition
$ python manage.py runserver
```
