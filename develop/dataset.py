import os
import argparse
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
import sys
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features

from config import *


DOC_PATH = 'alc_original/DOC/IS2011CHALLENGE'
DATA_PATH = 'alc_original'
TRAIN_TABLE = 'TRAIN.TBL'
D1_TABLE = 'D1.TBL'
D2_TABLE = 'D2.TBL'
TEST_TABLE = 'TESTMAPPING.txt'


class ALCDataset:
    def __init__(self, path):
        self.dataset_path = path
        self.__load_meta_file()

    def __process_meta(self, meta):
        meta['file_name'] = meta['file_name'].map(lambda x: x[x.find('/') + 1:].lower())
        meta['file_name'] = meta['file_name'].map(lambda x: x[:-8] + 'm' + x[-7:])
        meta['session'] = meta['file_name'].map(lambda x: x[:x.find('/')])
        meta['label'] = meta['user_state'].map(lambda x: 1 if x == 'I' else 0)
        return meta

    def __load_meta_file(self):
        assert os.path.exists(self.dataset_path)
        doc_folder = os.path.join(self.dataset_path, DOC_PATH)
        print(doc_folder)

        train_meta_path = os.path.join(doc_folder, TRAIN_TABLE)
        self.train_meta = pd.read_csv(train_meta_path, sep='\t', names=['file_name', 'bac', 'user_state'])
        self.train_meta = self.__process_meta(self.train_meta)

        d1_meta_path = os.path.join(doc_folder, D1_TABLE)
        self.d1_meta = pd.read_csv(d1_meta_path, sep='\t', names=['file_name', 'bac', 'user_state'])
        self.d1_meta = self.__process_meta(self.d1_meta)

        d2_meta_path = os.path.join(doc_folder, D2_TABLE)
        self.d2_meta = pd.read_csv(d2_meta_path, sep='\t', names=['file_name', 'bac', 'user_state'])
        self.d2_meta = self.__process_meta(self.d2_meta)

        test_meta_path = os.path.join(doc_folder, TEST_TABLE)
        self.test_meta = pd.read_csv(test_meta_path, sep='\t',
                                     names=['file_name', 'bac', 'user_state', 'test_file_name'])
        self.test_meta = self.test_meta[['file_name', 'bac', 'user_state']]
        self.test_meta = self.__process_meta(self.test_meta)
        
    def extract_opensmile_feature(self, split):
        split = split.lower()
        assert split in ('train', 'd1', 'd2', 'test')
        meta = getattr(self, f'{split}_meta')
        
        features = []
        for file_name in tqdm(meta['file_name']):
            wav_input_path = os.path.join(self.dataset_path, DATA_PATH, file_name)
            csv_output_path = "opensmile_feature.csv"
            if os.path.exists(csv_output_path):
                os.remove(csv_output_path)
            subprocess.run([OPENSMILE_PATH, "-C", OPENSMILE_CONF_PATH, "-I", wav_input_path, "-csvoutput", csv_output_path])
            feature = pd.read_csv(csv_output_path, delimiter=";").iloc[0, 2:].to_numpy()
            features.append(feature)                    
        features = np.stack(features)
        labels = meta['label'].to_numpy()
        
        if not os.path.exists(OPENSMILE_FEATURE_PATH):
            os.mkdir(OPENSMILE_FEATURE_PATH)
        np.save(os.path.join(OPENSMILE_FEATURE_PATH, f'{split}_x.npy'), features)
        np.save(os.path.join(OPENSMILE_FEATURE_PATH, f'{split}_y.npy'), labels)
            
        return features, labels
    
    def extract_surfboard_feature(self, split):
        split = split.lower()
        assert split in ('train', 'd1', 'd2', 'test')
        meta = getattr(self, f'{split}_meta')
        
        sounds = []
        for file_name in tqdm(meta['file_name']):
            sound = Waveform(path=os.path.join(self.dataset_path, DATA_PATH, file_name))
            sounds.append(sound)       
        features_df = extract_features(sounds, SURFBOARD_COMPONENTS, SURFBOARD_STATISTICS)
        features = features_df.to_numpy()
        labels = meta['label'].to_numpy()
        
        if not os.path.exists(SURFBOARD_FEATURE_PATH):
            os.makedirs(SURFBOARD_FEATURE_PATH)            
        np.save(os.path.join(SURFBOARD_FEATURE_PATH, f'{split}_x.npy'), features)
        np.save(os.path.join(SURFBOARD_FEATURE_PATH, f'{split}_y.npy'), labels)
            
        return features, labels
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for processing dataset')
    parser.add_argument('--toolbox', '-t', help='toolbox to extract features', default='surfboard')
    args = parser.parse_args()
    
    dataset = ALCDataset(DATASET_PATH)
    if args.toolbox == 'opensmile':
        print("Extracting opensmile features from train set...")
        feature_train, label_train = dataset.extract_opensmile_feature("train")
        print("Extracting opensmile features from dev1 set...")
        feature_d1, label_d1 = dataset.extract_opensmile_feature("d1")
        print("Extracting opensmile features from dev2 set...")
        feature_d2, label_d2 = dataset.extract_opensmile_feature("d2")
        print("Extracting opensmile features from test set...")
        feature_test, label_test = dataset.extract_opensmile_feature("test")
        print("Finished!")
    if args.toolbox == 'surfboard':
        print("Extracting surfboard features from train set...")
        feature_train, label_train = dataset.extract_surfboard_feature("train")
        print("Extracting surfboard features from dev1 set...")
        feature_d1, label_d1 = dataset.extract_surfboard_feature("d1")
        print("Extracting surfboard features from dev2 set...")
        feature_d2, label_d2 = dataset.extract_surfboard_feature("d2")
        print("Extracting surfboard features from test set...")
        feature_test, label_test = dataset.extract_surfboard_feature("test")
        print("Finished!")
