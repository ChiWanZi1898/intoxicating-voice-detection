import os
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
import sys

from config import *


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
        
    def extract_feature(self, split, conf_path):
        split = split.lower()
        assert split in ('train', 'd1', 'd2', 'test')
        meta = getattr(self, f'{split}_meta')
        
        features = []
        for file_name in tqdm(meta['file_name']):
            wav_input_path = os.path.join(self.dataset_path, DATA_PATH, file_name)
            csv_output_path = "feature.csv"
            if os.path.exists(csv_output_path):
                os.remove(csv_output_path)
            subprocess.run([OPENSMILE_PATH, "-C", conf_path, "-I", wav_input_path, "-csvoutput", csv_output_path])
            feature = pd.read_csv(csv_output_path, delimiter=";").iloc[0, 2:].to_numpy()
            features.append(feature)
                    
        features = np.stack(features)
        
        if not os.path.exists(FEATURE_PATH):
            os.mkdir(FEATURE_PATH)
        np.save(os.path.join(FEATURE_PATH, f'{split}_x.npy'), features)
            
        return features
    
    
if __name__ == "__main__":
    dataset = ALCDataset(DATASET_PATH)
    print("Extracting features from train set...")
    f_train = dataset.extract_feature("train", CONF_PATH)
    print("Extracting features from dev1 set...")
    f_d1 = dataset.extract_feature("d1", CONF_PATH)
    print("Extracting features from dev2 set...")
    f_d2 = dataset.extract_feature("d2", CONF_PATH)
    print("Extracting features from test set...")
    f_test = dataset.extract_feature("test", CONF_PATH)
    print("Finished!")
