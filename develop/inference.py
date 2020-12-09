import numpy as np
import pandas as pd
import os
import subprocess
from tqdm import tqdm
import sys
import argparse
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, plot_roc_curve
from imblearn.over_sampling import SMOTE
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features

from config import *
from train import ALCModel

plt.style.use('seaborn')


def get_opensmile_feature(wav_input_path):
    if os.path.exists("opensmile_feature.csv"):
        os.remove("opensmile_feature.csv")
    subprocess.run([OPENSMILE_PATH, "-C", OPENSMILE_CONF_PATH, "-I", wav_input_path, "-csvoutput", "opensmile_feature.csv"])
    feature = pd.read_csv("opensmile_feature.csv", delimiter=";").iloc[0, 2:].to_numpy()
    return feature

def get_surfboard_feature(wav_input_path, sample_rate):
    sound = Waveform(path=wav_input_path)
    feature_df = extract_features([sound], SURFBOARD_COMPONENTS, SURFBOARD_STATISTICS)
    feature = feature_df.iloc[0].to_numpy()
    return feature
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for model inference')
    parser.add_argument('--feature', '-f', help='feature extraction toolbox', default='surfboard')
    parser.add_argument('--model', '-M', help='machine learning model', default='svm')
    parser.add_argument('--input', '-I', help='input wav file', required=True)
    parser.add_argument('--sr', '-s', help='sample rate', default=16000)
    args = parser.parse_args()
    
    if args.feature == 'surfboard':
        model_path = SURFBOARD_MODEL_PATH
    if args.feature == 'opensmile':
        model_path = OPENSMILE_MODEL_PATH
    
    print('Loading model...')
    model = ALCModel(args.model)
    model.load_model(model_path)
    print('Finished!')
    
    print('Inference...')
    if args.feature == 'surfboard':
        feature = get_surfboard_feature(args.input, args.sr)
    if args.feature == 'opensmile':
        feature = get_opensmile_feature(args.input)
    prediction = model.predict([feature])
    print('Prediction result: ', prediction[0])

    