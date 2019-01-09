#!/usr/bin/env sh
node=0
ngpu=1

if [ "$node" = "1" ]
then
    node_ip="SH-IDC1-10-5-36-171"
else
    node_ip="SH-IDC1-10-5-36-172"
fi

echo $node_ip

cd /mnt/lustre/zhanshihan/sensetimework/caffe_torch/GenderClassification
source activate newpyenv


srun --mpi=pmi2 \
     --partition=Data \
     --gres=gpu:$ngpu \
     -n1 \
     -w $node_ip \
     --ntasks-per-node=1 \
     --job-name=IP0 \
     python main.py train  --use-gpu=True --env=classifier
     #nvidia-smi
     #pip install -r requirements.txt
     #python main.py help
     #python main.py test --train-data-root=./data/train --use-gpu=True --env=classifier
     #nvidia-smi

