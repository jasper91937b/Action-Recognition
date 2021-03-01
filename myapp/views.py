from PIL import Image
import requests
import base64
import io
from django.http import JsonResponse
from myapp.libs import MultiPersonClassifier
from myapp.libs import draw_result_img
from myapp.libs import get_the_skeleton_data_to_save_to_disk
from django.views.decorators.csrf import csrf_exempt  # import for csrf
from utils.lib_classifier import *  # Import all sklearn related libraries
from utils.lib_classifier import ClassifierOnlineTest
from utils.lib_tracker import Tracker
from utils.lib_openpose import SkeletonDetector
import utils.lib_commons as lib_commons
import utils.lib_plot as lib_plot
import utils.lib_images_io as lib_images_io
from django.shortcuts import render, redirect
from myapp.forms import UploadForm
from django.urls import reverse

import sys
import os
import numpy as np
import cv2
import time


# 找出根目錄Realtime-Action-Recognition-master與目前工作目錄，把根目錄加到環境變數中
ROOT = os.path.dirname(os.path.abspath(__file__)) + \
    "/Realtime-Action-Recognition-master/"
CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(ROOT)
sys.path.append(CURR_PATH)


# 將上傳的檔案寫入 static/upload
def handle_uploaded_file(f):
    with open('myapp/static/upload/'+f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


# 初始化參數，為了在瀏覽器顯示數據
list_result = None
dict_result = {}
# status_a = False # Alarm
# status_h = False # SOS
# status_o = False # opendoor
img_tmp_dict = {}
run_code = {}


@csrf_exempt
def predict(request, time=None):
    if time not in run_code.keys():
        run_code[time] = True
    if request.method == 'POST':
        upload = UploadForm(request.POST, request.FILES)
        if upload.is_valid():
            videoinput = request.FILES['file']
            handle_uploaded_file(videoinput)
    else:
        upload = UploadForm()
        return render(request, "index.html", {'form': upload})

    # def get_folder_name(data_type):
    #     '''
    #     根據data_type和data_path計算輸出文件夾名稱。
    #     該腳本的最終輸出如下所示：
    #     DST_FOLDER/folder_name/video.avi
    #     DST_FOLDER/folder_name/skeletons/XXXXX.txt
    #     '''
    #     if data_type == "video":  # /root/data/video.avi --> video
    #         folder_name = videoinput.name.split(".")[-2]
    #     return folder_name

    DATA_TYPE = "video"
    DATA_PATH = f"{CURR_PATH}/static/upload/{videoinput.name}"
    MODEL_PATH = f"{ROOT}/model/dnn_model.h5"
    # DST_FOLDER_NAME = get_folder_name(DATA_TYPE)
    DST_FOLDER_NAME = videoinput.name.split(".")[-2]
    output_folder = f"{ROOT}/output"

    # ----設定

    cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
    cfg = cfg_all["s5_test.py"]

    CLASSES = np.array(cfg_all["classes"])
    # "{:05d}.txt"
    SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

    # 動作識別：用於提取特徵的幀數。
    WINDOW_SIZE = int(cfg_all["features"]["window_size"])  # 5

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
    OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]  # cmu
    OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]  # 656x368

    # Display 設定
    img_disp_desired_rows = int(
        cfg["settings"]["display"]["desired_rows"])  # 480

    # -- Detector, tracker, classifier
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(
        MODEL_PATH, CLASSES, WINDOW_SIZE)

    # -- Image reader and displayer
    images_loader = lib_images_io.ReadFromVideo(
        DATA_PATH,
        sample_interval=VIDEO_SAMPLE_INTERVAL)

    # 網頁上不顯示
    # img_displayer = lib_images_io.ImageDisplayer()
    # cv2.namedWindow('cv2_display_window', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('cv2_display_window',570,460)
    # cv2.moveWindow("cv2_display_window", 900, 275)

    # -- Init output
    # output folder
    os.makedirs(DST_FOLDER, exist_ok=True)
    os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

    # video writer
    # video_writer = lib_images_io.VideoWriter(
    #     DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)

    # -- Read images and process
    try:
        ith_img = -1
        while images_loader.has_image() and run_code[time]:

            # -- Read image
            img = images_loader.read_image()
            ith_img += 1
            img_disp = img.copy()
            print(f"\nProcessing {ith_img}th image ...")

            # -- Detect skeletons
            humans = skeleton_detector.detect(img)
            skeletons, scale_h = skeleton_detector.humans_to_skels_list(
                humans)  # skeletons shape : (1, 36)

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
                                       skeleton_detector, multiperson_classifier, img_disp_desired_rows, dict_id2label, scale_h)
            # 將BGR照片轉為RGB
            img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
            # -- Display image, and write to video.avi
            
            # 網頁上不顯示
            # img_displayer.display(img_disp, wait_key_ms=1)

            global img_tmp_dict
            img_tmp_dict[time] = img_rgb
            # video_writer.write(img_disp)

            # -- Get skeleton data and save to file
            skels_to_save = get_the_skeleton_data_to_save_to_disk(
                dict_id2skeleton, dict_id2label)

            result = np.array(skels_to_save)

            global list_result
            list_result = {i+1: result[i, 1] for i in range(result.shape[0])}
            print(list_result)

            # save the result to txt files for all frame
            lib_commons.save_listlist(
                DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                SKELETON_FILENAME_FORMAT.format(ith_img),
                skels_to_save)

            # global dict_result
            # global status_a
            # global status_h
            # global status_o

            # for uid, action in list_result.items():
            #     if uid not in dict_result.keys():
            #         dict_result[uid] = []
            #         dict_result[uid].append(action)
            #     else:
            #         if len(dict_result[uid]) > 15:
            #             dict_result[uid].pop(0)
            #             dict_result[uid].append(action)
            #         else:
            #             dict_result[uid].append(action)
            #     count_kick = dict_result[uid].count("kick")
            #     count_punch = dict_result[uid].count("punch")
            #     count_sos = dict_result[uid].count("sos")
            #     count_opendoor = dict_result[uid].count("opendoor")

            #     if count_kick > 10 or count_punch > 10:
            #         status_a = True
            #         lineNotifyMessage(msg="發現疑似違規行為")
            #         dict_result[uid].clear()
            #     else :
            #         status_a = False

            #     if count_sos > 10:
            #         status_h = True
            #         lineNotifyMessage(msg="有人發出求救訊號")
            #         dict_result[uid].clear()
            #     else :
            #         status_h = False

            #     if count_opendoor > 10:
            #         status_o = True
            #         dict_result[uid].clear()
            #         lineNotifyMessage(msg="請檢查隨身物品是否攜帶")
            #     else :
            #         status_o = False
        return render(request, "index.html", {'form': upload})

    except Exception as e:
        print(e)
    finally:
        # video_writer.stop()
        print("Program ends")
        list_result = None


# 把資料顯示在網頁上


def send_data(request):
    result = list_result
    return JsonResponse({"data": result, "showa": status_a, "showh": status_h, "showo": status_o})


# Line notify
def lineNotifyMessage(msg):

    headers = {
        "Authorization": "Bearer UPDl3lVjMjwqH1uBoIjFA1fqKaZnBaRoKP6oQ1pcoqd",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    params = {"message": msg}

    r = requests.post("https://notify-api.line.me/api/notify",
                      headers=headers, params=params)
    # 呼叫LINE Notify的API


indexs = 0
imagelist = []
filepath = os.path.join(os.path.dirname(__file__), 'static/img/loading2.gif')
imgs = Image.open(filepath)
for i in range(imgs.n_frames-1):
    imgs.resize((640, 480))
    imagelist.append(imgs.copy())
    imgs.seek(i+1)


def img(request, time):
    if time not in img_tmp_dict.keys():
        global indexs
        if indexs > len(imagelist):
            indexs = 0
        im = imagelist[indexs]
        indexs += 1

    else:
        img = img_tmp_dict[time]
        im = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)
    base64_bytes = base64.b64encode(rawBytes.read())
    base64_string = base64_bytes.decode("utf-8")
    return JsonResponse({"images": base64_string})


def stop_predict(request, time):
    global run_code
    run_code[time] = False
    return redirect('/action/')
