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

plt.style.use('seaborn')


class ALCModel:
    def __init__(self, method):
        self.method = method
        
        if method == 'logistic':
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            self.clf = LogisticRegression(penalty='l1', solver='saga', C=1.0, n_jobs=8)
        elif method == 'neighbor':
            # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
            self.clf = KNeighborsClassifier(n_neighbors=5, leaf_size=30, n_jobs=8)
        elif method == 'linear_svm':
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
            self.clf = LinearSVC(C=0.1)
        elif method == 'svm':
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            self.clf = SVC(C=10.0, kernel='rbf')
        elif method == 'forest':
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            self.clf = RandomForestClassifier(n_estimators=200, n_jobs=8)
        elif method == 'adaboost':
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            self.clf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
        elif method == 'gradboost':
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
            self.clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
        else:
            raise NotImplementedError
        
    def fit(self, x, y):
        self.clf.fit(x, y)
    
    def predict(self, x):
        prediction = self.clf.predict(x)
        return prediction
    
    def evaluate(self, x, label, roc=True):
        pred = self.predict(x)
        acc = np.mean(pred == label)
        report = sklearn.metrics.classification_report(label, pred)
        if roc:
            curve = plot_roc_curve(self.clf, x, label)
            plt.show()
        return acc, report
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)        
        filename = os.path.join(path, '{}.pkl'.format(self.method))
        pickle.dump(self.clf, open(filename, "wb"))
        
    def load_model(self, path):
        filename = os.path.join(path, '{}.pkl'.format(self.method))
        self.clf = pickle.load(open(filename, "rb"))


def load_data(feature='surfboard', method='full', dimension=50, use_dev=True, balance=True):
    assert feature in ['surfboard', 'opensmile']
    assert dim_reduct in ['full', 'pca', 'ica']
    
    if feature == 'surfboard':
        feature_path = SURFBOARD_FEATURE_PATH
    else:
        feature_path = OPENSMILE_FEATURE_PATH
    
    train_x = np.load(os.path.join(feature_path, 'train_x.npy'), allow_pickle=True)
    train_y = np.load(os.path.join(feature_path, 'train_y.npy'), allow_pickle=True)
    test_x = np.load(os.path.join(feature_path, 'test_x.npy'), allow_pickle=True)
    test_y = np.load(os.path.join(feature_path, 'test_y.npy'), allow_pickle=True)
    
    if use_dev:
        dev1_x = np.load(os.path.join(feature_path, 'd1_x.npy'), allow_pickle=True)
        dev1_y = np.load(os.path.join(feature_path, 'd1_y.npy'), allow_pickle=True)
        dev2_x = np.load(os.path.join(feature_path, 'd2_x.npy'), allow_pickle=True)
        dev2_y = np.load(os.path.join(feature_path, 'd2_y.npy'), allow_pickle=True)
        train_x = np.concatenate([train_x, dev1_x, dev2_x])
        train_y = np.concatenate([train_y, dev1_y, dev2_y])
        
    if balance:
        smote = SMOTE(random_state=0)
        balance_x, balance_y = smote.fit_resample(train_x, train_y)
        
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    if balance:
        balance_x = scaler.transform(balance_x)
    test_x = scaler.transform(test_x)
    
    if method == 'pca':
        pca = PCA(n_components=dimension)
        pca.fit(train_x)
        train_x = pca.transform(train_x)
        if balance:
            balance_x = pca.transform(balance_x)
        test_x = pca.transform(test_x)
    
    if method == 'ica':
        ica = FastICA(n_components=dimension)
        ica.fit(train_x)
        train_x = ica.transform(train_x)
        if balance:
            balance_x = ica.transform(balance_x)
        test_x = ica.transform(test_x)
        
    if balance:
        return balance_x, balance_y, test_x, test_y
    return train_x, trian_y, test_x, test_y
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for model training')
    parser.add_argument('--feature', '-f', help='feature extraction toolbox', default='surfboard')
    parser.add_argument('--model', '-M', help='machine learning model', default='svm')
    parser.add_argument('--method', '-m', help='dimension reduction method', default='full')
    parser.add_argument('--dim', '-d', help='dimension after PCA or ICA', default=50)
    parser.add_argument('--usedev', '-u', help='whether use dev set for training or net', default=True)
    parser.add_argument('--balance', '-b', help='whether use SMOTE to balance data', default=True)
    args = parser.parse_args()
    
    print('Loading data...')
    train_data, trian_label, test_data, test_label = load_data(feature=args.feature,
                                                               method=args.method, 
                                                               dimension=args.dim, 
                                                               use_dev=args.usedev, 
                                                               balance=args.balance)
    print('Finished!')
    
    print('Training model...')
    model = ALCModel(args.model)
    model.fit(train_data, trian_label)
    print('Finished!')
    
    print('Saving model...')
    if args.feature == 'surfboard':
        model.save_model(SURFBOARD_MODEL_PATH)
    if args.feature == 'opensmile':
        model.save_model(OPENSMILE_MODEL_PATH)
    print('Finished!')
    
    print('Classification report on test data:')
    acc, report = model.evaluate(test_data, test_label, roc=True)    
    print(report)  
    