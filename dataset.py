import pandas as pd
import os
import librosa
from queue import Queue
from threading import Thread
from typing import Tuple

DOC_PATH = 'alc_original/DOC/IS2011CHALLENGE'
DATA_PATH = 'alc_original'
TRAIN_TABLE = 'TRAIN.TBL'
D1_TABLE = 'D1.TBL'
D2_TABLE = 'D2.TBL'
TEST_TABLE = 'TESTMAPPING.txt'
SR = 16000


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
        """Load meta file.

        :return: None
        """
        assert os.path.exists(self.dataset_path)
        doc_folder = os.path.join(self.dataset_path, DOC_PATH)

        train_meta_path = os.path.join(doc_folder, TRAIN_TABLE)
        self.train_meta = pd.read_csv(train_meta_path, sep='\t', names=['file_name', 'bac', 'user_state'])
        self.train_meta = self.__process_meta(self.train_meta)
        # just read first 20% lines
        # self.train_meta = self.train_meta.iloc[:len(self.train_meta)//5,:]

        d1_meta_path = os.path.join(doc_folder, D1_TABLE)
        self.d1_meta = pd.read_csv(d1_meta_path, sep='\t', names=['file_name', 'bac', 'user_state'])
        self.d1_meta = self.__process_meta(self.d1_meta)
        # just read first 20% lines
        # self.d1_meta = self.d1_meta.iloc[:len(self.d1_meta)//5,:]

        d2_meta_path = os.path.join(doc_folder, D2_TABLE)
        self.d2_meta = pd.read_csv(d2_meta_path, sep='\t', names=['file_name', 'bac', 'user_state'])
        self.d2_meta = self.__process_meta(self.d2_meta)
        # just read first 20% lines
        # self.d2_meta = self.d2_meta.iloc[:len(self.d2_meta)//5,:]

        test_meta_path = os.path.join(doc_folder, TEST_TABLE)
        self.test_meta = pd.read_csv(test_meta_path, sep='\t',
                                     names=['file_name', 'bac', 'user_state', 'test_file_name'])
        self.test_meta = self.test_meta[['file_name', 'bac', 'user_state']]
        self.test_meta = self.__process_meta(self.test_meta)
        # just read first 20% lines
        # self.test_meta = self.test_meta.iloc[:len(self.test_meta)//5,:]

    def __load_wav(self, path):
        audio, _ = librosa.load(path, sr=SR)
        return audio

    def load_data(self, split, num_threads=4):
        split = split.lower()
        assert split in ('train', 'd1', 'd2', 'test')
        meta = getattr(self, f'{split}_meta')
        audios_list = [{} for _ in range(num_threads)]
        q = Queue()

        def load(q, audios, dataset_path, data_path, i):
            while not q.empty():
                if i == 0:
                    print(f'{q.qsize():05d} left.', end='\r')
                path = q.get()
                audio_path = os.path.join(dataset_path, data_path, path)
                audios[path] = self.__load_wav(audio_path)
                q.task_done()
            return True

        for file_name in meta['file_name']:
            q.put(file_name)

        for i in range(num_threads):
            worker = Thread(target=load, args=(q, audios_list[i], self.dataset_path, DATA_PATH, i))
            worker.setDaemon(True)
            worker.start()

        q.join()
        audios = {}
        for i in range(num_threads):
            audios.update(audios_list[i])

        data = []
        for file_name in meta['file_name']:
            data.append(audios[file_name])

        return data, meta

    def slice_data(self, data: list, meta, k: int) -> Tuple[list, list]:
        sliced_data = list()
        sliced_meta = list()
        d_len = k * SR
        j = 0
        for line in data:
            times = len(line) // d_len
            for i in range(times):
                tmp = line[(i * d_len) : ((i + 1) * d_len)]
                sliced_data.append(tmp)
                sliced_meta.append(meta['label'][j])
            j += 1
        return sliced_data, sliced_meta
