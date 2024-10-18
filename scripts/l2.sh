export CUDA_VISIBLE_DEVICES=0

# L1: Arabic,     Speaker: ABA/   SKA/   YBAA/  ZHAA/
# L1: Mandarin,   Speaker: BWC/   LXC/   NCC/   TXHC/
# L1: Hindi,      Speaker: ASI/   RRBI/  SVBI/  TNI/ 
# L1: Korean,     Speaker: HJK/   HKK/   YDCK/  YKWK/
# L1: Spanish,    Speaker: EBVS/  ERMS/  MBMPS/ NJS/
# L1: Vietnamese, Speaker: HQTV/  PNV/   THV/   TLV/                

dataset_dir=/path/to/l2_arctic/
model_dir=facebook/wav2vec2-base-960h
dataset_name=l2arctic

# parameter for CEA
lr1=2e-5
steps1=10
# parameter for STCR
lr2=2e-4
steps2=10

# specify L1
l1=arabic
# then specify speaker
speaker_list="ABA SKA YBAA ZHAA"

for speaker in $speaker_list; do
    python main_cea.py --model_base $model_dir \
                    --steps1 $steps1 \
                    --steps2 $steps2 \
                    --lr1 $lr1 \
                    --lr2 $lr2 \
                    --dataset_name $dataset_name \
                    --dataset_dir $dataset_dir/$speaker \
                    --log_dir exps_l2/exps_l1_${l1}_$speaker 
done


