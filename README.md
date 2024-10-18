# Advancing Test-Time Adaptation in Wild Acoustic Test Settings

This repository contains code and analysis for the paper [Advancing Test-Time Adaptation in Wild Acoustic Test Settings](https://arxiv.org/abs/2310.09505).

## Overview

Acoustic foundation models, fine-tuned for Automatic Speech Recognition (ASR), suffer from performance degradation in wild acoustic test settings when deployed in real-world scenarios. Stabilizing online Test-Time Adaptation (TTA) under these conditions remains an open and unexplored question. Existing wild vision TTA methods often fail to handle speech data effectively due to the unique characteristics of high-entropy speech frames, which are unreliably filtered out even when containing crucial semantic content. Furthermore, unlike static vision data, speech signals follow short-term consistency, requiring specialized adaptation strategies. In this work, we propose a novel wild acoustic TTA method tailored for ASR fine-tuned acoustic foundation models. Our method, Confidence-Enhanced Adaptation, performs frame-level adaptation using a confidence-aware weight scheme to avoid filtering out essential information in high-entropy frames. Additionally, we apply consistency regularization during test-time optimization to leverage the inherent short-term consistency of speech signals. Our experiments on both synthetic and real-world datasets demonstrate that our approach outperforms existing baselines under various wild acoustic test settings, including Gaussian noise, environmental sounds, accent variations, and sung speech.

## Quick Start


### Installation

```bash
cd CEA
pip install -r requirements.txt
```

### Data Source
- [Librispeech](https://www.openslr.org/12)/
- [L2-Arctic](https://psi.engr.tamu.edu/l2-arctic-corpus/)/
- [DSing](https://github.com/groadabike/Kaldi-Dsing-task)/


### Synthesize Data

To obtained the synthesized dataset LS-P, run the following scripts

```bash
python noisyspeech_synthesizer.py
```

### Run CEA 

Our main code include `./main_cea.py`, `./utils.py`. 

To run experiments on different datasets, use scripts from `scripts/` 

#### LS-C 
```bash
bash ./scripts/ls-c.sh
```

#### LS-P
```bash
bash ./scripts/ls-p.sh
```

#### L2-Arctic 
```bash
bash ./scripts/l2.sh
```

#### Singing (DSing-dev, DSing-test, Hansen)
```bash
bash ./scripts/dsing.sh
```

See more details about the parameters in each script.


## Acknowledgements 

This repository is based on [Tent](https://github.com/DequanWang/tent), [SAR](https://github.com/mr-eggplant/SAR/), and [SUTA](https://github.com/DanielLin94144/Test-time-adaptation-ASR-SUTA) and [SGEM](https://github.com/drumpt/SGEM).

