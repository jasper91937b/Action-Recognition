from django.shortcuts import render, redirect
import numpy as np
import cv2

# write the upload file to upload folder 
def handle_uploaded_file(f):  
    with open('myapp/static/upload/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)
            
from myapp.forms import UploadForm
import sys
import os

# find Root and current path, add Root to path
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/Realtime-Action-Recognition-master/"
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+ "/"
sys.path.append(ROOT)

import time
import numpy as np
import cv2
import utils.lib_images_io as lib_images_io
import utils.lib_plot as lib_plot
import utils.lib_commons as lib_commons
from utils.lib_openpose import SkeletonDetector
from utils.lib_tracker import Tracker
from utils.lib_classifier import ClassifierOnlineTest
from utils.lib_classifier import *  # Import all sklearn related libraries
from django.views.decorators.csrf import csrf_exempt # import for csrf


# init the global variable videoinput, show data in browser needed. 
list_result = None
dict_result = {} 
status_a = False # Alarm
status_h = False # SOS
status_o = False # opendoor

@csrf_exempt
def predict(request):
    if request.method == 'POST':  
        upload = UploadForm(request.POST, request.FILES)  
        if upload.is_valid():
            videoinput = request.FILES['file']
            handle_uploaded_file(videoinput)
    else:  
        upload  = UploadForm()  
        return render(request,"main.html",{'form':upload }) 


    def get_folder_name(data_type):
        ''' 
        根據data_type和data_path計算輸出文件夾名稱。
        該腳本的最終輸出如下所示：
        DST_FOLDER/folder_name/video.avi
        DST_FOLDER/folder_name/skeletons/XXXXX.txt
        '''
        if data_type == "video":  # /root/data/video.avi --> video
            folder_name = videoinput.name.split(".")[-2]
        return folder_name
    

    
    DATA_TYPE = "video" 
    DATA_PATH = f"{CURR_PATH}/static/upload/{videoinput.name}" 
    MODEL_PATH = f"{ROOT}/model/dnn_model.h5" 
    DST_FOLDER_NAME = get_folder_name(DATA_TYPE)
    output_folder = f"{ROOT}/output"

    # ----設定

    cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
    cfg = cfg_all["s5_test.py"]

    CLASSES = np.array(cfg_all["classes"])
    SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"] # "{:05d}.txt"

    # 動作識別：用於提取特徵的幀數。
    WINDOW_SIZE = int(cfg_all["features"]["window_size"]) # 5

    # Output folder
    DST_FOLDER = output_folder + "/" + DST_FOLDER_NAME + "/"
    DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
    DST_VIDEO_NAME = cfg["output"]["video_name"]

    # 輸出video.avi的幀率
    DST_VIDEO_FPS = float(cfg["output"]["video_fps"])

    # Video 設定
    # 如果data_type為video，則設置採樣間隔。
    # 例如，如果為3，則video的讀取速度將提高3倍。
    VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                    ["video_sample_interval"])

    # Openpose 設定
    OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"] # cmu
    OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"] # 656x368

    # Display 設定
    img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"]) # 480

    # -- Function
    class MultiPersonClassifier(object):
        ''' 
        This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
        用於識別多人的動作。
        dict_id2clf => {id : ClassifierOnlineTest object}
        id2label => {id : predict action}
        '''
        def __init__(self, model_path, classes):
            self.dict_id2clf = {}  

            # Define a function for creating classifier for new people.
            self._create_classifier = lambda human_id: ClassifierOnlineTest(
                model_path, classes, WINDOW_SIZE, human_id)

        def classify(self, dict_id2skeleton):
            ''' 
            Classify the action type of each skeleton in dict_id2skeleton 
            '''
            
            # Predict each person's action
            id2label = {} 
            for id, skeleton in dict_id2skeleton.items():
                
                # if id not in dict_id2clf,means this is a new person
                # add this new person
                
                if id not in self.dict_id2clf:  
                    self.dict_id2clf[id] = self._create_classifier(id)
                    
                classifier = self.dict_id2clf[id]
                id2label[id] = classifier.predict(skeleton)  # predict label
                # print("\n\nPredicting label for human{}".format(id))
                # print("  skeleton: {}".format(skeleton))
                # print(" label: {}".format(id2label[id]))
            return id2label

        def get_classifier(self, id):
            ''' 
            Get the classifier based on the person id.
            Arguments:
                id {int or "min"}
            '''
            if len(self.dict_id2clf) == 0:
                return None
            # if id==min, use the person have lower id
            if id == 'min':
                id = min(self.dict_id2clf.keys())
            return self.dict_id2clf[id]

    def draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                        skeleton_detector, multiperson_classifier):
        ''' 
        Draw skeletons, labels, and prediction scores onto image for display 
        img_disp => copy of the input image
        '''

        # Resize to a proper size for display
        r, c = img_disp.shape[0:2] # find the size of image, height 480 and width 640 
        desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
        
        # resize to 640*480  => unchange
        img_disp = cv2.resize(img_disp,
                            dsize=(desired_cols, img_disp_desired_rows))

        # Draw all people's skeleton
        skeleton_detector.draw(img_disp, humans)

        # Draw bounding box and label of each person
        if len(dict_id2skeleton):
            for id, label in dict_id2label.items():
                skeleton = dict_id2skeleton[id]
                # scale the y data back to original
                skeleton[1::2] = skeleton[1::2] / scale_h # y had multiply scale_h when record the skeleton
                lib_plot.draw_action_result(img_disp, id, skeleton, label)
        
        # add text "fps"
        cv2.putText(img_disp, "Frame:" + str(ith_img),
                    (10, 25), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                    color=(0, 0, 0), thickness=2)

        # Draw predicting score for only 1 person
        if len(dict_id2skeleton):
            classifier_of_a_person = multiperson_classifier.get_classifier(
                id='min')
            classifier_of_a_person.draw_scores_onto_image(img_disp)
        return img_disp


    def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
        '''
        In each image, for each skeleton, save the:
            human_id, label, and the skeleton positions of length 18*2.
        So the total length per row is 2+36=38
        '''
        skels_to_save = []
        for human_id in dict_id2skeleton.keys():
            label = dict_id2label[human_id]
            skeleton = dict_id2skeleton[human_id]
            skels_to_save.append([human_id, label] + skeleton.tolist())
        return skels_to_save

 
    # -- Detector, tracker, classifier
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(MODEL_PATH, CLASSES)

    # -- Image reader and displayer
    images_loader = lib_images_io.ReadFromVideo(
                DATA_PATH,
                sample_interval=VIDEO_SAMPLE_INTERVAL)
    img_displayer = lib_images_io.ImageDisplayer()

    cv2.namedWindow('cv2_display_window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cv2_display_window',800,700)
    cv2.moveWindow("cv2_display_window", 900, 450)
    
    # -- Init output

    # output folder
    os.makedirs(DST_FOLDER, exist_ok=True)
    os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

    # video writer
    video_writer = lib_images_io.VideoWriter(
        DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)

    # -- Read images and process
    try:
        ith_img = -1
        while images_loader.has_image():

            # -- Read image
            img = images_loader.read_image()
            ith_img += 1
            img_disp = img.copy()
            print(f"\nProcessing {ith_img}th image ...")

            # -- Detect skeletons
            humans = skeleton_detector.detect(img)
            skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
            # skeletons shape : (1, 36)
            skeletons = skeletons

            # -- Track people
            # dict_id2skeleton => {id : skeletons}
            dict_id2skeleton = multiperson_tracker.track(
                skeletons)
            
            # -- Recognize action of each person
            
            if len(dict_id2skeleton):
                dict_id2label = multiperson_classifier.classify(
                    dict_id2skeleton)

            # -- Draw
            img_disp = draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                                        skeleton_detector, multiperson_classifier)
        
            # # Print label of a person
            # if len(dict_id2skeleton):
            #     min_id = min(dict_id2skeleton.keys())
            #     print("prediced label is :", dict_id2label[min_id])

            # -- Display image, and write to video.avi
            img_displayer.display(img_disp, wait_key_ms=1)
            video_writer.write(img_disp)

            # -- Get skeleton data and save to file
            skels_to_save = get_the_skeleton_data_to_save_to_disk(
                dict_id2skeleton)

            result = np.array(skels_to_save)
            global list_result
            list_result = { i+1:result[i,1] for i in range(result.shape[0])}
            print(list_result)
            
            # save the result to txt files for all frame
            lib_commons.save_listlist(
                DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                SKELETON_FILENAME_FORMAT.format(ith_img),
                skels_to_save)

            global dict_result  
            global status_a
            global status_h
            global status_o
            
            for uid, action in list_result.items():
                if uid not in dict_result.keys():
                    dict_result[uid] = []
                    dict_result[uid].append(action)
                else:
                    if len(dict_result[uid]) > 15:
                        dict_result[uid].pop(0)
                        dict_result[uid].append(action)
                    else:
                        dict_result[uid].append(action) 
                count_kick = dict_result[uid].count("kick")
                count_punch = dict_result[uid].count("punch")
                count_sos = dict_result[uid].count("sos")
                count_opendoor = dict_result[uid].count("opendoor")
                if count_kick > 10 or count_punch > 10:
                    # print(f"ALARM {uid}!")
                    status_a = True
                    lineNotifyMessage(msg="發現疑似違規行為")
                    dict_result[uid].clear()
                else :
                    status_a = False
                
                if count_sos > 10:
                    status_h = True
                    # print(f"SOS {uid}!")
                    lineNotifyMessage(msg="有人發出求救訊號")
                    dict_result[uid].clear()
                else :
                    status_h = False
                
                if count_opendoor > 10:
                    status_o = True
                    # print(f"Check {uid}!")
                    dict_result[uid].clear()
                    lineNotifyMessage(msg="請檢查隨身物品是否攜帶")
                    # return redirect("http://localhost:8000/item/up")
                else :
                    status_o = False
    
        return render(request,"main.html",{'form':upload })    
    except Exception as e:
        print(e)
    finally:
        video_writer.stop()
        print("Program ends")
        list_result = None

# show data in browser
from django.http import JsonResponse

def send_data(request):
    result = list_result
    # if status_h:
    #     lineNotifyMessage(msg="有人發出求救訊號")
    # if status_a:
    #     lineNotifyMessage(msg="發現疑似違規行為")
    # # if status_o:
    # #     lineNotifyMessage(msg="請檢查隨身物品是否攜帶")
    
    return JsonResponse({"data":result, "showa": status_a, "showh": status_h, "showo":status_o})


import requests
# Line notify
def lineNotifyMessage(msg):

    headers = {
        "Authorization": "Bearer UPDl3lVjMjwqH1uBoIjFA1fqKaZnBaRoKP6oQ1pcoqd",
        "Content-Type": "application/x-www-form-urlencoded"
    }
 
    params = {"message":msg}
 
    r = requests.post("https://notify-api.line.me/api/notify",  
                      headers=headers, params=params) 
    # 呼叫LINE Notify的API





  

        
