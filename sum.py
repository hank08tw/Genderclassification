import sleep
for i in range(20):
    python main.py test --test-data-root=data/test2 --load-model-path='checkpoints/resnet34_0822_06:38:22.pth' --batch-size=16 --model='ResNet34' --num-workers=4
