import os 
import torch
import argparse
from tqdm import tqdm
from jiwer import wer

from data import load_dataset
from utils import *
from transformers import Wav2Vec2ForCTC, HubertForCTC, WavLMForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    parser = argparse.ArgumentParser(description="WildSpeechTTA")
    parser.add_argument('--model_base', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps1', type=int, default=10)
    parser.add_argument('--steps2', type=int, default=0)
    parser.add_argument('--lr1', type=float, default=2e-5)
    parser.add_argument('--lr2', type=float, default=2e-4)
    parser.add_argument('--lm', action='store_true')
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/path/to/LibriSpeech')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tc_coef', type=float, default=0.3)
    parser.add_argument('--em_coef', type=float, default=0.3)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--log_dir', type=str, default='./exps')
 
    # args for LS 
    parser.add_argument('--noise_type', type=str, default='')
    parser.add_argument('--noise_level', type=int, default=0)

    args = parser.parse_args()

    model_base = args.model_base

    steps_1 = args.steps1
    steps_2 = args.steps2
    lr1 = args.lr1
    lr2 = args.lr2
    episodic = args.episodic
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    tc_coef = args.tc_coef
    em_coef = args.em_coef
    temp =  args.temp
    log_dir = args.log_dir
    noise_type = args.noise_type
    noise_level = args.noise_level

    skip_short_thd = None

    
    exp_name = dataset_name+'_'+model_base.split('/')[-1]+'_'+str(lr1)+'_'+str(lr2)+'_'+str(steps_1)+'_'+str(steps_2)+'_'+'_noiselevel_'+str(noise_level)+'_noisetype_'+str(noise_type)
   
    set_seed(42)

    # load data
    dataset = load_dataset(dataset_name, dataset_dir, batch_size, noise_type, noise_level)

    print('------------------------------------')
    print(f'exp: {exp_name}')
    print(f'lr1 = {lr1}')
    print(f'lr2 = {lr2}')
    print(f'step1 = {steps_1}')
    print(f'step2 = {steps_2}')

    print(f'noise_type = {noise_type}')
    print(f'noise_level = {noise_level}')

    # setup 
    if args.lm:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_base, sampling_rate=SAMPLE_RATE, return_attention_mask=True)

    if 'wav2vec2' in model_base:
        model = Wav2Vec2ForCTC.from_pretrained(model_base).eval().cuda()  
    elif 'hubert' in model_base:
        model = HubertForCTC.from_pretrained(model_base).eval().cuda()  
    elif 'wavlm' in model_base:
        model = WavLMForCTC.from_pretrained(model_base).eval().cuda()      

    model = configure_model(model)
    params1, param_names1 = collect_params(model, True, False, True)
    params2, param_names2 = collect_params(model, False, False, True)

    params = [params1, params2]
    lrs = [lr1, lr2]
    optimizer = setup_optimizer(params, lrs)

    if episodic: 
        model_state, optimizer_state = copy_model_and_optimizer(model, optimizer)

    # adapt stats
    no_adapt_count = 0
    show_text = False
    gt_texts = []
    ori_transcriptions = []
    transcription_dict = {}
    durations = []
    werrs = []

    # customized
    to_display = [steps_1+steps_2-1]

    for i in to_display:
        transcription_dict[i] = []

    for batch in tqdm(dataset):
        lens, wavs, texts, files = batch
        inputs = processor(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.cuda()
        
        duration = input_values.shape[1] / SAMPLE_RATE
        durations.append(duration)
        
        if episodic: 
            model, optimizer = load_model_and_optimizer(model, optimizer, model_state, optimizer_state)
        
        # unadapted / source model  
        with torch.no_grad():
            outputs = model(input_values).logits
        ori_transcription = decode(outputs, processor, args.lm)
        ori_transcriptions += ori_transcription
        ori_wer = wer(list(texts), list(ori_transcription))
        print("Original WER: ", ori_wer)

        # skip short 
        if skip_short_thd is not None: 
            if outputs.shape[1] <= skip_short_thd:
                print(f'Do not adapt since length is {outputs.shape[1]}')
                no_adapt_count += 1
                continue
        
        # adaptation
        for i in range(steps_1+steps_2): 
            outputs = forward_and_adapt(input_values, model, model_base, optimizer, tc_coef, em_coef, i, steps_1, temp)
            if i in to_display:
                transcription = decode(outputs, processor, args.lm)
                ada_wer = wer(list(texts), list(transcription))
                print("Adapt-{} WER:  {}".format(i+1, ada_wer))
                if show_text:
                    print('Ground Truth: ', texts, '\nTranscription: ', transcription)
                if i == to_display[-1]:
                    werr = ori_wer - ada_wer
                    werrs.append(werr)
                transcription_dict[i] += transcription
        
        del input_values
        torch.cuda.empty_cache()
        gt_texts += texts

    print("Model Base: ", model_base)
    print("Non-adapted Count = ", no_adapt_count)
    print("Dataset Num = ", len(dataset))
    print("Original WER:", wer(gt_texts, ori_transcriptions))
    
    wer_dict = {}
    for i, trans in transcription_dict.items():
        wer_dict[i] = wer(gt_texts, trans)
        print("TTA-{} WER: {}".format(i+1, wer_dict[i]))


    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with open(os.path.join(log_dir, exp_name), 'w') as f: 
        f.write(f"Original WER: {wer(gt_texts, ori_transcriptions)}\n")
        for i, wer_res in wer_dict.items():
            f.write(f"TTA-{i+1} WER: {wer_res}\n")
 
        f.write(f'model = {model_base}\n')
        f.write(f'lr1 = {lr1}\n')
        f.write(f'lr2 = {lr2}\n')
        f.write(f'step1 = {steps_1}\n')
        f.write(f'step2 = {steps_2}\n')
        f.write(f'noise_type = {noise_type}\n')
        f.write(f'noise_level = {noise_level}\n')
    
