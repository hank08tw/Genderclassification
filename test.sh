#!/bin/bash  
#!/bin/bash 
  
for i in 1 2 3 4 5  
 do  
  python main.py test --test-data-root=data/test2 --load-model-path='checkpoints/resnet34_0822_06:38:22.pth' --batch-size=16 --model='ResNet34' --num-workers=1
 done
