import re
import pandas as pd 
from builtins import str as unicode
from torch.utils.data import Dataset            


class DSingDataset(Dataset):
    def __init__(self, name, bucket_size, path, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        
        data_dict = {
            'dsing-dev': '/dev.csv',
            'dsing': '/test.csv',
            'hansen': '/Hansen.csv',
            'jamendo': '/Jamendo.csv', 
            'mauch': '/Mauch.csv'
        }
        
        data_path = path + data_dict[name]

        df = pd.read_csv(data_path, sep=',')
        text = df['wrd'].values
        file_list = df['wav'].values
        durations = df['duration'].values

        print(len(file_list), len(text))
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt, dur in sorted(zip(file_list, text, durations), reverse=not ascending, key=lambda x:float(x[-1]))])

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
