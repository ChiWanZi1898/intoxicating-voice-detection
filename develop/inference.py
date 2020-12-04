import numpy as np
import os
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

from config import *
from train import ALCModel

plt.style.use('seaborn')


def get_feature(wav_input_path, csv_output_path):
    if os.path.exists(csv_output_path):
        os.remove(csv_output_path)
    subprocess.run([OPENSMILE_PATH, "-C", CONF_PATH, "-I", wav_input_path, "-csvoutput", csv_output_path])
    feature = pd.read_csv(csv_output_path, delimiter=";").iloc[0, 2:].to_numpy()
    return feature
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for model training')
    parser.add_argument('--model', '-M', help='machine learning model', default='svm')
    parser.add_argument('--input', '-I', help='input wav file', required=True)
    parser.add_argument('--output', '-o', help='output csv file', default='feature.csv')
    args = parser.parse_args()
    
    print('Loading model...')
    model = ALCModel(args.model)
    model.load_model(MODEL_PATH)
    print('Finished!')
    
    print('Inference...')
    feature = get_feature(args.input, args.ouput)
    prediction = model.predict([feature])
    print('Prediction result: ', prediction[0])

    