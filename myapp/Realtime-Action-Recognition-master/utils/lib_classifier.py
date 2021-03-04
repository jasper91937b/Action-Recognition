
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# -- Settings
NUM_FEATURES_FROM_PCA = 50

if True:
    import sys
    import os
    from utils.lib_feature_proc import FeatureGenerator
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    sys.path.append(ROOT)
    SRC_PROCESSED_FEATURES = f"{ROOT}/data_proc/features_X.csv"
    SRC_PROCESSED_FEATURES_LABELS = f"{ROOT}/data_proc/features_Y.csv"
    
    # Read Data for PCA
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)  # features
    Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)  # labels

    # PCA fit
    n_components = min(NUM_FEATURES_FROM_PCA, X.shape[1])
    pca = PCA(n_components=50, whiten=True)
    pca.fit(X)  



# -- Classes
class ClassifierOfflineTrain(object):
    def __init__(self):
        self._model_name = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(50,), kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.Dropout(0.3),
                layers.Dense(7, activation='softmax')
            ])

    # Train model
    def train(self, X, Y):
        # PCA transform
        print("Before PCA, X.shape = ", X.shape)
        self.pca = pca
        print("Sum eig values:", np.sum(self.pca.explained_variance_ratio_))
        X_new = self.pca.transform(X)
        print("After PCA, X.shape = ", X_new.shape)

        self._model_name.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy']) 
        print(self._model_name.summary())
        history = self._model_name.fit(X_new, Y, batch_size=100, epochs=20, validation_split=0.1, verbose=2) 
        
        # print accuracy plot
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['training', 'validation'], loc='lower right')
        plt.show()
        
        # print loss plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training', 'validation'], loc='upper right')
        plt.show()
 
    
    def predict(self, X):
        ''' 
        Predict the class
        '''
        Y_predict = self._model_name.predict(self.pca.transform(X))
        Y_predict = np.argmax(axis = 1)
        return Y_predict
    
    def predict_and_evaluate(self, te_X, te_Y):
        ''' 
        Test model on test set and obtain accuracy 
        '''
        te_Y_predict = self._model_name.predict(self.pca.transform(te_X))
        te_Y_predict = te_Y_predict.argmax(axis = 1)
        N = len(te_Y)
        n = sum(te_Y_predict == te_Y)
        accu = n / N
        return accu, te_Y_predict
    
    def _predict_proba(self, X):
        Y_probs = self._model_name.predict(self.pca.transform(X))
        return Y_probs
    
    def save_model(self, path):
        self._model_name.save(path) 



class ClassifierOnlineTest(object):
    ''' 
    Classifier for online inference.
        The input data to this classifier is the raw skeleton data, so they
            are processed by `class FeatureGenerator` before sending to the
            self.model trained by `class ClassifierOfflineTrain`. 
    '''

    def __init__(self, model_path, action_labels, window_size, human_id=0):
        # -- Settings
        self.human_id = human_id

        # load model
        self.model = keras.models.load_model(model_path)
        if self.model is None:
            print("my Error: failed to load model")
            assert False

        self.action_labels = action_labels
        self.THRESHOLD_SCORE_FOR_DISP = 0.5

        # -- Time serials storage
        self.feature_generator = FeatureGenerator(window_size)
        self.reset()

    def reset(self):
        self.feature_generator.reset()
        self.scores_hist = deque()
        self.scores = None

    def predict(self, skeleton):
        ''' 
        Predict the class (string) of the input raw skeleton 
        '''
        LABEL_UNKNOWN = ""
        is_features_good, features = self.feature_generator.add_cur_skeleton(
            skeleton)
        
        if is_features_good:
            # convert to 2d array
            features = features.reshape(-1, features.shape[0])

            # PCA
            features = pca.transform(features)
            curr_scores = self.model.predict(features)[0]
            
            self.scores = self.smooth_scores(curr_scores)

            if self.scores.max() < self.THRESHOLD_SCORE_FOR_DISP:  # If lower than threshold, bad
                prediced_label = LABEL_UNKNOWN
            else:
                predicted_idx = self.scores.argmax()
                prediced_label = self.action_labels[predicted_idx]
        else:
            prediced_label = LABEL_UNKNOWN
        return prediced_label

    def smooth_scores(self, curr_scores):
        ''' 
        Smooth the current prediction score
        by taking the average with previous scores
        '''
        self.scores_hist.append(curr_scores)
        DEQUE_MAX_SIZE = 2
        if len(self.scores_hist) > DEQUE_MAX_SIZE:
            self.scores_hist.popleft()

        if 1:  # Use sum
            score_sums = np.zeros((len(self.action_labels),))
            for score in self.scores_hist:
                score_sums += score
            score_sums /= len(self.scores_hist)
            print("\nMean score:\n", score_sums)
            return score_sums

        else:  # Use multiply
            score_mul = np.ones((len(self.action_labels),))
            for score in self.scores_hist:
                score_mul *= score
            return score_mul

    def draw_scores_onto_image(self, img_disp):
        if self.scores is None:
            return

        for i in range(0, len(self.action_labels)):

            FONT_SIZE = 0.5
            TXT_X = 20
            TXT_Y = 100 + i*30
            COLOR_INTENSITY = 255

            
            label = self.action_labels[i]
            s = "{:<5}: {:.2f}".format(label, self.scores[i])
            COLOR_INTENSITY *= (0.0 + 1.0 * self.scores[i])**0.5

            # cv2.putText(img_disp, text=s, org=(TXT_X, TXT_Y),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SIZE,
            #             color=(0, 0, int(COLOR_INTENSITY)), thickness=1)
