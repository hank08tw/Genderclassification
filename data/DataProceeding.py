import random
import shutil,os
import glob
#from PIL import Image

female_dir = '/home/zhangjw/Work/AcdamicDesign/Code/Gender_classification/venv/data/test/female'
male_dir = '/home/zhangjw/Work/AcdamicDesign/Code/Gender_classification/venv/data/test/male'

test_dir = "/home/zhangjw/Work/GenderClassification/data/test1"
train_dir = "/home/zhangjw/Work/GenderClassification/data/train"


list_female = os.listdir(female_dir) #列出文件夹下所有的目录与文件

for i in range(0,len(list_female)):
    shutil.copy((female_dir + '/' + str(list_female[i])), test_dir + '/' + 'female.' + str(i) + '.jpg')

list_male = os.listdir(male_dir)
for i in range(0,len(list_male)):
    shutil.copy((male_dir + '/' + str(list_male[i])), test_dir + '/' + 'male.' + str(i) + '.jpg')




