export CUDA_VISIBLE_DEVICES=0

dataset_dir=/path/to/LibriSpeech

# specify the model base
model_dir=facebook/wav2vec2-base-960h
#model_dir=facebook/wav2vec2-large-960h-lv60-self
#model_dir=danieleV9H/hubert-base-libri-clean-ft100h
#model_dir=facebook/hubert-large-ls960-ft
#model_dir=patrickvonplaten/wavlm-libri-clean-100h-base-plus
#model_dir=patrickvonplaten/wavlm-libri-clean-100h-large

dataset_name=librispeech
# parameter for CEA
lr1=2e-5
steps1=10
# parameter for STCR
lr2=2e-4
steps2=10

# specify the noise level ranging from 1 to 5, default 1
noise_level=1

python main_cea.py --model_base $model_dir \
                --steps1 $steps1 \
                --steps2 $steps2 \
                --lr1 $lr1 \
                --lr2 $lr2 \
                --dataset_name $dataset_name \
                --dataset_dir $dataset_dir \
                --log_dir exps_ls_c \
                --noise_type gaussian \
                --noise_level $noise_level
                