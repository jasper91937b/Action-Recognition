import cv2
import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/Realtime-Action-Recognition-master/"
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+ "/"
sys.path.append(ROOT)
sys.path.append(CURR_PATH)
import utils.lib_plot as lib_plot
from utils.lib_classifier import ClassifierOnlineTest
# 多人動作偵測 ㄐecognizing actions of multiple people

class MultiPersonClassifier(object):
    ''' 
    用於識別多人的動作。
    dict_id2clf => {id : ClassifierOnlineTest object}
    id2label => {id : predict action}
    '''
    def __init__(self, model_path, classes, WINDOW_SIZE):
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
        # if id==min, 找id最小的人
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]

def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton, dict_id2label):
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

def draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                        skeleton_detector, multiperson_classifier, img_disp_desired_rows, dict_id2label, scale_h):
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
                skeleton[1::2] = skeleton[1::2] / scale_h # y 在擷取特徵的時候有乘scale_h，要除回來
                lib_plot.draw_action_result(img_disp, id, skeleton, label)
        
        # add text "fps"
        cv2.putText(img_disp, "Frame:" + str(ith_img),
                    (10, 25), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                    color=(0, 0, 0), thickness=2)

        # Draw predicting score for only 1 person
        # 取最小的id
        if len(dict_id2skeleton):
            classifier_of_a_person = multiperson_classifier.get_classifier(
                id='min')
            classifier_of_a_person.draw_scores_onto_image(img_disp)
        return img_disp