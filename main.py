#coding:utf8
from config import opt
import os
import torch as t
import models#old
#import torchvision.models as models#new
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer as vis
from tqdm import tqdm
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt

def Visualizer(**kwargs):
    #opt.parse(kwargs)
    # import ipdb;
    # ipdb.set_trace()
    # configure model
    model = getattr(models, opt.model)().eval()
    #model=models.resnet18(pretrained=True)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data
    test_data = DogCat(opt.visualize_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # print(test_dataloader)
    print("come")
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data, volatile=True)
        # print("data is {}".format(data))
        # print("path is {}".format(path))
        imgs = input
        if opt.use_gpu: input = input.cuda()
        import time
        print("time is ")
        print(time.time())
        score = model(input)
        print(time.time())
        # print("score is :{}".format(score))
        label = score.max(dim=1)[1].data.tolist()
        print("Label is {}".format(label))
        img = torchvision.utils.make_grid(imgs)
        img = img.numpy().transpose(1, 2, 0)
        std = [0.5, 0.5, 0.5]
        mean = [0.5, 0.5, 0.5]
        img = img * std + mean
        # print(img)
        plt.imshow(img)
        plt.show()



'''
def test(**kwargs):
    opt.parse(kwargs)
    # import ipdb;
    # ipdb.set_trace()
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    #print(test_dataloader)
    print("come")
    results = []
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data, volatile = True)
        #print("data is {}".format(data))
        print("path is {}".format(path))
        if opt.use_gpu: input = input.cuda()
        import time
        print("time is ")
        print(time.time())
        score = model(input)
        print(time.time())
        print("score is :{}".format(score))
        probability = t.nn.functional.softmax(score)[:,0].data.tolist()
        print("probability is :".format(probability))
        label = score.max(dim = 1)[1].data.tolist()
        print("Label is {}".format(label))
        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]

        results += batch_results
    write_csv(results,opt.result_file)
    print(results)
    return results
'''

def test(**kwargs):
    # opt.parse(kwargs)
    # import ipdb;
    # ipdb.set_trace()
    # configure model
    #model = getattr(models, opt.model)().eval()
    model=models.resnet18(pretrained=True)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data
    test_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    #print(test_dataloader)
    print("come")
    results = []
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data,volatile = True)
        #print("data is {}".format(data))
        print("path is {}".format(path))
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        print("score is :{}".format(score))
        probability = t.nn.functional.softmax(score)[:,0].data.tolist()
        print("probability is :".format(probability))
        label = score.max(dim = 1)[1].data.tolist()
        print("Label is {}".format(label))
        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]
        results += batch_results
    write_csv(results,opt.result_file)
    print(results)
    val_data = DogCat(opt.test_data_root,train=False)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    val_cm, val_accuracy = val(model, val_dataloader)

    print("end val function")
    print(val_cm.value(),val_accuracy)
    print("test function end")
    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)
    
def train(**kwargs):
    #opt.parse(kwargs)
    vis = Visualizer()

    # step1: configure model
    print("come step 1")
    #model = getattr(models, opt.model)()
    model=models.resnet18(pretrained=True)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: data
    print("come here step 2")
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    print(train_dataloader)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    print("come step 3")
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
        
    # step4: meters
    print("come step 4")
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch):
        print("start training")
        loss_meter.reset()
        confusion_matrix.reset()
        # print("original matrix is:{}".format(confusion_matrix.value()))
        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            print("label is {}".format(label))
            # train model 
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            print("score is ",score)
            print("target is",target)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            #print("Epoch is :{s},Loss is {}".format(epoch,loss))
            
            
            # meters update and visualize
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)

            #if ii%opt.print_freq==opt.print_freq-1:
            #     vis.plot('loss', loss_meter.value()[0])
            #
            #     # 进入debug模式
            #     if os.path.exists(opt.debug_file):
            #         import ipdb;
            #         ipdb.set_trace()
        print("Now learning rate is {}".format(lr))
        if (epoch % opt.lr_decay_epoch == 0):
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    model.save()

    # validate and visualize

    val_cm, val_accuracy = val(model, val_dataloader)

    print("end val function")
    print(val_cm.value(),val_accuracy)
    vis.plot('val_accuracy',val_accuracy)
    vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))

        # update learning rate

    if loss_meter.value()[0] > previous_loss:
        lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # previous_loss = loss_meter.value()[0]

def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    print("start val function")
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        print("val")
        print(label)
        val_input = Variable(input, volatile=True)
        print("come val")
        if opt.use_gpu:
            val_input = val_input.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))
    print("val finish")
    print("matrix is :{}".format(confusion_matrix.value()))
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    print("accuracy is {}".format(accuracy))
    return confusion_matrix, accuracy







def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
    #train()
