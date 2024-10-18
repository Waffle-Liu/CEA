import os
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


def read_text(file):
    src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    idx = file.split('/')[-1].split('.')[0]

    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]


class LibriDataset(Dataset):
    def __init__(self, bucket_size, path, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        split = ['test-other']
        # List all wave files
        file_list = []
        for s in split: 
            split_list = list(Path(os.path.join(path, s)).rglob("*.flac"))
            file_list += split_list
        
        text = []
        for f in tqdm(file_list, desc='Read text'):
            transcription = read_text(str(f))
            text.append(transcription)

        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)


class NoisyLibriDataset(Dataset):
    def __init__(self, bucket_size, path, noise_type=None, noise_level=None, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        split = ['test-other']
        noise_level_mapping = {
            1: 10,
            2: 5,
            3: 0,
            4: -5,
            5: -10
        }

        # List all wave files
        file_list = []
        for s in split: 
            split_list = list(Path(os.path.join(path, s)).rglob("*.flac"))
            file_list += split_list
        file_list.sort()

        text = []
        for f in tqdm(file_list, desc='Read text'):
            transcription = read_text(str(f))
            text.append(transcription)
        
        noise_snr = noise_level_mapping[noise_level]
        print(noise_type, noise_snr)
        if noise_type:
            snr_string = f"_{noise_snr}.0" if int(noise_snr) in [10, 5, 0, -5, -10] else ""
            file_list = sorted(list(Path(os.path.join(path, f"/data1/hongfu/data/noisyLibriSpeech/libri_test_noise_{snr_string}/{noise_type}")).rglob("*.wav")))

        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)