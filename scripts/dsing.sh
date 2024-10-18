export CUDA_VISIBLE_DEVICES=0

# specify the dataset
#dataset_dir=/path/to/DSing/dev
dataset_dir=/path/to/DSing/test
#dataset_dir=/path/to/Hansen

# specify the model base
model_dir=facebook/wav2vec2-base-960h
#model_dir=facebook/wav2vec2-large-960h-lv60-self
#model_dir=danieleV9H/hubert-base-libri-clean-ft100h
#model_dir=facebook/hubert-large-ls960-ft
#model_dir=patrickvonplaten/wavlm-libri-clean-100h-base-plus
#model_dir=patrickvonplaten/wavlm-libri-clean-100h-large

#dataset_name=DSing-dev
dataset_name=DSing
#dataset_name=Hansen
# parameter for CEA
lr1=2e-5
steps1=10
# parameter for STCR
lr2=2e-4
steps2=10


python main_cea.py --model_base $model_dir \
                --steps1 $steps1 \
                --steps2 $steps2 \
                --lr1 $lr1 \
                --lr2 $lr2 \
                --dataset_name $dataset_name \
                --dataset_dir $dataset_dir \
                --log_dir exps_$dataset_name 