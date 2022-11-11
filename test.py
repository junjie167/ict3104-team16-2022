from __future__ import division
import time
import os
import argparse
import sys
import torch
import csv
import cv2


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-model', type=str, default='')
parser.add_argument('-APtype', type=str, default='wap')
parser.add_argument('-randomseed', type=str, default='False')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-num_channel', type=str, default='False')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-kernelsize', type=str, default='False')
parser.add_argument('-feat', type=str, default='False')
parser.add_argument('-split_setting', type=str, default='CS')
parser.add_argument('-video', type=str, default='undefined')
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# set random seed
if args.randomseed=="False":
    SEED = 0
elif args.randomseed=="True":
    SEED = random.randint(1, 100000)
else:
    SEED = int(args.randomseed)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# print('Random_SEED!!!:', SEED)

from torch.optim import lr_scheduler
from torch.autograd import Variable

import json

import pickle
import math


if str(args.APtype) == 'map':
    from apmeter import APMeter


batch_size = int(args.batch_size)

if args.dataset == 'TSU':
    split_setting=str(args.split_setting)
    
    from smarthome_i3d_per_video import TSU as Dataset
    from smarthome_i3d_per_video import TSU_collate_fn as collate_fn
    classes=51
    
    if split_setting =='CS':
        train_split = './data/smarthome_CS_51.json'
        test_split = './data/smarthome_CS_51.json'
        
    elif split_setting =='CV':
        train_split = './data/smarthome_CV_51.json'
        test_split = './data/smarthome_CV_51.json'
    
    rgb_root = 'C:/Users/ngjun/3104/TSU/TSUProj/pipline/data/RGB_i3d_16frames_64000_SSD'
    skeleton_root='/skeleton/feat/Path/' # 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data_rgb_skeleton(train_split, val_split, root_skeleton, root_rgb):
    # Load Data
   
    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root_skeleton, root_rgb, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn) # 8
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root_skeleton, root_rgb, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn) #2

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


def load_data(train_split, val_split, root):
    # Load Data
  
    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    since = time.time()

    best_map = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            # train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch)
            probs.append(prob_val)
            sched.step(val_loss)

            if best_map < val_map:
                best_map = val_map
                torch.save(model.state_dict(),'./'+str(args.model)+'/weight_epoch_'+str(args.lr)+'_'+str(epoch))
                torch.save(model,'./'+str(args.model)+'/model_epoch_'+str(args.lr)+'_'+str(epoch))
                print('save here:','./'+str(args.model)+'/weight_epoch_'+str(args.lr)+'_'+str(epoch))

def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, :int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)
    
    outputs_final = activation

    if args.model=="PDAN":
        # print('outputs_final1', outputs_final.size())
        outputs_final = outputs_final[:,0,:,:]
    # print('outputs_final',outputs_final.size())
    outputs_final = outputs_final.permute(0, 2, 1)  
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)  

    loss = loss_f 

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data
        loss.backward()
        optimizer.step()
    if args.APtype == 'wap':
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    print('train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss

def getNumFrames(videoName):
    cap = cv2.VideoCapture("./data/video/" + videoName)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def output_csvResults(activityIndexes,numFrames,val_map,actAccList_tnsr):
    file = open('./results/result.csv', 'w', encoding='UTF8', newline='')
    # file = open('./data/generatedAnnotations/{model_name}_{video_name}.csv'.format(model_name=model, video_name=video), 'w', encoding='UTF8', newline='')
    activityArr = ["Enter", "Walk","Make_coffee", "Get_water", "Make_coffee", "Use_Drawer", "Make_coffee.Pour_grains", 
    "Use_telephone", "Leave", "Put_something_on_table", "Take_something_off_table" , "Pour.From_kettle", 
    "Stir_coffee/tea", "Drink.From_cup", "Dump_in_trash", "Make_tea", "Make_tea.Boil_water", "Use_cupboard",
    "Make_tea.Insert_tea_bag" , "Read", "Take_pills", "Use_fridge", "Clean_dishes", "Clean_dishes.Put_something_in_sink",
    "Eat_snack", "Sit_down", "Watch_TV", "Use_laptop", "Get_up", "Drink.From_bottle", "Pour.From_bottle",
    "Drink.From_glass", "Lay_down", "Drink.From_can", "Write", "Breakfast", "Breakfast.Spread_jam_or_butter",
    "Breakfast.Cut_bread", "Breakfast.Eat_at_table", "Breakfast.Take_ham", "Clean_dishes.Dry_up", "Wipe_table",
    "Cook", "Cook.Cut", "Cook.Use_stove", "Cook.Stir", "Cook.Use_oven", "Clean_dishes.Clean_with_water",
    "Use_tablet", "Use_glasses", "Pour.From_can"]    
    writer = csv.writer(file)  
    # Write Rows for Model Precision 
    writer.writerow(["Mean Average Precision of Model: ", val_map.item()])
    writer.writerow([])
    acc_dict = {}
    actAccList = actAccList_tnsr.numpy()
    writer.writerow(["Activity","Average Class Prediction"])
    for i in range(0, len(actAccList)):
        if (actAccList[i] > 0):
            acc_dict[activityArr[i]] = actAccList[i]
            writer.writerow([activityArr[i], actAccList[i]])

    # Write Rows for Training Section (loss, epoch,etc?)
    writer.writerow([])
    writer.writerow(["Trained on","Train m-AP","Tested on","Prediction m-AP"])
    # To be done

    # Write Rows Video Frame data
    writer.writerow([])
    writer.writerow(["Event","Start_frame","End_frame","Video_Name","Prediction Accuracy for the video"])
    # To be done
    currentFrames = 0
    endFrames = 0
    framesPerIndex = numFrames/len(activityIndexes)
    videoName = args.video.replace(".mp4","")
    
    for i in range(0, len(activityIndexes)):
        if ((i < len(activityIndexes)-1) and (activityIndexes[i] == activityIndexes[i+1]) ):
            endFrames += framesPerIndex
        else:
            endFrames += framesPerIndex
            new_row = []
            for action in acc_dict.keys():
                if (activityArr[activityIndexes[i]] == action):
                    new_row = [activityArr[activityIndexes[i]], (currentFrames), (endFrames),videoName,acc_dict[action]]
                    break
                else:
                    new_row = [activityArr[activityIndexes[i]], (currentFrames),videoName, (endFrames),0.0]
            writer.writerow(new_row)
            currentFrames = endFrames

    file.close()
    print("csv file output to /result folder")
    return
    
def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        activityIndexes = []
        for i in range(1, len(probs.data.cpu().numpy()[0])):
            activityIndexes.append(np.argmax(probs.data.cpu().numpy()[0][i]))

        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter


    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print('val-map:', val_map)
    activityAcc = 100 * apm.value()

    apm.reset()

    return full_probs, epoch_loss, val_map, activityAcc,activityIndexes


if __name__ == '__main__':
    print(str(args.model))
    print('batch_size:', batch_size)
    print('cuda_avail', torch.cuda.is_available())

    if args.mode == 'flow':
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'skeleton':
        dataloaders, datasets = load_data(train_split, test_split, skeleton_root)
    elif args.mode == 'rgb':
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    model = torch.load(args.load_model)
    prob_val, val_loss, val_map, actAccuracy,activityIndexes = val_step(model, 0, dataloaders['val'], 0) 
    numFrames = getNumFrames(args.video)
    # print("numframes",numFrames)
    output_csvResults(activityIndexes,numFrames,val_map,actAccuracy)
    
    


