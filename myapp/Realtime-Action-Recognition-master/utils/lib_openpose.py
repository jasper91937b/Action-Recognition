'''
This script defines the class `SkeletonDetector`
`SkeletonDetector` => 從每一張圖片偵測人的骨骼
which is used for detecting human skeleton from image.
The code is copied and modified from src/githubs/tf-pose-estimation
'''

# -- Libraries
if True: # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"    # Realtime-Action-Recognition-master資料夾
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"  # utils資料夾
    sys.path.append(ROOT)

import sys, os, time, argparse, logging
import cv2
# openpose packages
sys.path.append(ROOT + "src/githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common


# -- Settings 設定
MAX_FRACTION_OF_GPU_TO_USE = 0.4
IS_DRAW_FPS = True

# -- Helper functions輔助功能
# logging module，設定basicConfig
def _set_logger():
    logger = logging.getLogger('TfPoseEstimator')
    logger.setLevel(logging.DEBUG)
    logging_stream_handler = logging.StreamHandler()
    logging_stream_handler.setLevel(logging.DEBUG)
    logging_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logging_stream_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_stream_handler)
    return logger

def _set_config():
    ''' 
    Set the max GPU memory to use 
    設定GPU memory
    '''
    # For tf 1.13.1, The following setting is needed
    import tensorflow as tf
    from tensorflow import keras
    # for tensorflow 2
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU_TO_USE
    return config


def _get_input_img_size_from_string(image_size_str):
    ''' 
    If input image_size_str is "656x368", then output (656, 368)
    如果image_size設定 656x368，會先使用 x 做 split，再取整數  
    '''
    width, height = map(int, image_size_str.split('x'))

    # 寬跟高必須是16的倍數
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)



# -- Main class

class SkeletonDetector(object):
    # This class is mainly copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model="cmu", image_size="432x368"):
        ''' 
        Arguments:
            model {str}: "cmu" or "mobilenet_thin".        
            image_size {str}: resize input images before they are processed. 
                Recommends : 432x368, 336x288, 304x240, 656x368,
        使用cmu, 656x368 
        '''
        # -- Check input
        assert(model in ["mobilenet_thin", "cmu"])
        self._w, self._h = _get_input_img_size_from_string(image_size)
        
        # -- Set up openpose model
        self._model = model
        self._resize_out_ratio = 4.0 # Resize heatmaps before they are post-processed. If image_size is small, this should be large.
        self._config = _set_config() # GPU setting
        # TfPoseEstimator class 寫在tf_pose.estimator內
        self._tf_pose_estimator = TfPoseEstimator(
            get_graph_path(self._model), 
            target_size=(self._w, self._h),
            tf_config=self._config)
        self._prev_t = time.time()
        self._cnt_image = 0
        
        # -- Set logger
        self._logger = _set_logger()
        

    def detect(self, image):
        ''' 
        Detect human skeleton from image.
        Arguments:
            image: RGB image with arbitrary size. It will be resized to (self._w, self._h).
            把圖像 640 x 480 RGB (640, 480, 3) 會resize成 656, 368
        Returns:
            humans {list of class Human}: 
                `class Human` is defined in 
                "src/githubs/tf-pose-estimation/tf_pose/estimator.py"
                
                The variable `humans` is returned by the function
                `TfPoseEstimator.inference` which is defined in
                `src/githubs/tf-pose-estimation/tf_pose/estimator.py`.

                I've written a function `self.humans_to_skels_list` to 
                extract the skeleton from this `class Human`.

                [BodyPart:0-(0.47, 0.29) score=0.71 BodyPart:1-(0.46, 0.38) score=0.84 BodyPart:2-(0.42, 0.38) score=0.72 BodyPart:3-(0.40, 0.48) score=0.78 BodyPart:4-(0.39, 0.57) score=0.71 BodyPart:5-(0.51, 0.38) score=0.69 BodyPart:6-(0.52, 0.48) score=0.80 BodyPart:7-(0.54, 0.58) score=0.74 BodyPart:8-(0.44, 0.56) score=0.70 BodyPart:9-(0.45, 0.70) score=0.87 BodyPart:10-(0.46, 0.83) score=0.68 BodyPart:11-(0.50, 0.55) score=0.65 BodyPart:12-(0.53, 0.67) score=0.83 BodyPart:13-(0.52, 0.81) score=0.73 BodyPart:14-(0.46, 0.28) score=0.71 BodyPart:15-(0.48, 0.28) score=0.80 BodyPart:16-(0.44, 0.29) score=0.73 BodyPart:17-(0.49, 0.29) score=0.76]
        '''
        
        self._cnt_image += 1
        # 讀取第一張照片的高跟寬
        if self._cnt_image == 1:
            self._image_h = image.shape[0]
            self._image_w = image.shape[1]
            # scale_h => 1 * 高 / 寬 
            self._scale_h = 1.0 * self._image_h / self._image_w # 480/640 => 0.75
        t = time.time()

        # Do inference
        humans = self._tf_pose_estimator.inference(
            image, resize_to_default=(self._w > 0 and self._h > 0),
            upsample_size=self._resize_out_ratio)

        # Print result and time cost
        # 計算花了多久時間
        elapsed = time.time() - t
        self._logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        ''' 
        Draw human skeleton on img_disp inplace.
        把骨骼畫在照片上
        Argument:
            img_disp {RGB image}
            humans {a class returned by self.detect}
        '''
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        if IS_DRAW_FPS:
            cv2.putText(img_disp,
                        "fps: {:.1f}".format( (1.0 / (time.time() - self._prev_t) )),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)
        self._prev_t = time.time()

    def humans_to_skels_list(self, humans, scale_h = None): 
        ''' Get skeleton data of (x, y * scale_h) from humans.
        Arguments:
            humans {a class returned by self.detect}
            scale_h {float}: scale each skeleton's y coordinate (height) value.
                Default: (image_height / image_widht).
        Returns:
            skeletons {list of list}: a list of skeleton.
                [[0.4695121951219512, 0.22010869565217395, 0.46646341463414637, 0.28125, 0.42073170731707316, 0.28125, 0.4024390243902439, 0.3586956521739131, 0.39634146341463417, 0.42391304347826086, 0.5091463414634146, 0.28532608695652173, 0.5274390243902439, 0.36277173913043476, 0.5396341463414634, 0.4361413043478261, 0.4420731707317073, 0.41983695652173914, 0.45121951219512196, 0.5217391304347826, 0.4573170731707317, 0.6236413043478262, 0.4969512195121951, 0.4157608695652174, 0.5274390243902439, 0.5135869565217391, 0.5121951219512195, 0.6154891304347826, 0.4603658536585366, 0.2078804347826087, 0.47560975609756095, 0.2078804347826087, 0.4451219512195122, 0.21603260869565216, 0.49085365853658536, 0.22010869565217395]]

                Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
                總共有18個骨骼，每個有x, y 座標，總共36個值
            scale_h {float}: The resultant height(y coordinate) range.
                The x coordinate is between [0, 1].
                The y coordinate is between [0, scale_h]
        '''
        if scale_h is None:
            scale_h = self._scale_h
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                # BodyPart:0-(0.47, 0.29) score=0.71 => part_idx 0 , x 0.47, y 0.29, score 0.71
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_h # 每個y都乘了一個scale_h，之後要除回來
            skeletons.append(skeleton)
        return skeletons, scale_h
    

def test_openpose_on_webcamera():
    # -- Initialize web camera reader
    from utils.lib_images_io import ReadFromWebcam, ImageDisplayer
    webcam_reader = ReadFromWebcam(max_framerate=10)
    img_displayer = ImageDisplayer()
    
    # -- Initialize openpose detector    
    skeleton_detector = SkeletonDetector("mobilenet_thin", "432x368")

    # -- Read image and detect
    import itertools
    for i in itertools.count():
        img = webcam_reader.read_image()
        if img is None:
            break
        print(f"Read {i}th image...")

        # Detect
        humans = skeleton_detector.detect(img)
        
        # Draw
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp)
        
    print("Program ends")

if __name__ == "__main__":
    test_openpose_on_webcamera()
