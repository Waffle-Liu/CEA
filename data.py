import torch
torch.manual_seed(0)
import torchaudio
from functools import partial
from torch.utils.data import DataLoader
import numpy as np

SAMPLE_RATE = 16000

def collect_audio_batch(batch, extra_noise=0., maxLen=600000):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...]
    '''
    def audio_reader(filepath):
        
        wav, sample_rate = torchaudio.load(filepath)
        if sample_rate != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(wav)
        wav = wav.reshape(-1)
        if wav.shape[-1] >= maxLen:
            print(f'{filepath} has len {wav.shape}, truncate to {maxLen}')
            wav = wav[:maxLen]
            print(wav.shape)
        wav += extra_noise * torch.randn_like(wav)
        
        return wav

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            feat = audio_reader(str(b[0])).numpy()
            file.append(str(b[0]).split('/')[-1].split('.')[0])
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(b[1])

    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])

    return audio_len, audio_feat, text, file


def create_dataset(name, path, batch_size=1, noise_type=None, noise_level=None):

    print("Dataset: ",name)
    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    elif name.lower() == "noisylibrispeech":
        from corpus.librispeech import NoisyLibriDataset as Dataset
    elif name.lower() == "chime":
        from corpus.CHiME import CHiMEDataset as Dataset
    elif name.lower() == "l2arctic":
        from corpus.l2arctic import L2ArcticDataset as Dataset
    elif name.lower() in ['dsing', 'dsing-dev', 'hansen', 'jamendo', 'mauch']:
        from corpus.DSing import DSingDataset as Dataset
    else:
        raise NotImplementedError

    loader_bs = batch_size
    
    if name.lower() in ['dsing', 'dsing-dev', 'hansen', 'jamendo', 'mauch']:
        dataset = Dataset(name.lower(), batch_size, path)
    elif name.lower() == "l2arctic":
        dataset = Dataset(batch_size, path)
    elif name.lower() == 'noisylibrispeech':
        dataset = Dataset(batch_size, path, noise_type=noise_type, noise_level=noise_level)
    else: 
        dataset = Dataset(batch_size, path)
    print(f'[INFO]    There are {len(dataset)} samples.')

    return dataset, loader_bs


def load_dataset(name='librispeech', path=None, batch_size=1, noise_type='', noise_level=0, num_workers=0):

    dataset, loader_bs = create_dataset(name, path, batch_size, noise_type, noise_level)
    if noise_type in ['gaussian']:
        gaussian_level_mapping = {
            1: 0.005,
            2: 0.01,
            3: 0.015,
            4: 0.02,
            5: 0.03
        }
        extra_noise = gaussian_level_mapping[noise_level]
    else:
        extra_noise = 0.

    collate_fn = partial(collect_audio_batch, extra_noise=extra_noise)
    
    def _init_fn(worker_id):
        np.random.seed(42)

    dataloader = DataLoader(dataset, batch_size=loader_bs, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, worker_init_fn=_init_fn)
    
    return dataloader
