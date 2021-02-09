import numpy as np
import torch
from torch.autograd import grad

if __name__ == "__main__":
    print(torch.zeros(4))
    weight_all = torch.zeros([2, 2])
    weight_1 = torch.tensor([1, 2])
    weight_2 = torch.tensor([3, 4])
    weight_all[0] = weight_1
    weight_all[1] = weight_2
    print(weight_all)
    # weight_1 = torch.tensor([5, 6])
    weight_1[0] = 5
    print(weight_all)
    print(weight_1)


import os
meragefiledir = os.getcwd() + '\\noRats_neg'
filenames = os.listdir(meragefiledir)
size = len(filenames)
file1 = open('train.txt', 'w')
file2 = open('val.txt', 'w')
file3 = open('test.txt', 'w')

i = 0
for filename in filenames:
    filepath = meragefiledir + '\\' + filename
    for line in open(filepath):
        if i <= int(size*0.6):
            file1.writelines(line.replace("\n", "") + '\t' + "0" + '\n')
        elif i <= int(size*0.8):
            file2.writelines(line.replace("\n", "") + '\t' + "0" + '\n')
        else:
            file3.writelines(line.replace("\n", "") + '\t' + "0" + '\n')
        i += 1


i = 0
meragefiledir = os.getcwd() + '\\noRats_pos'
filenames = os.listdir(meragefiledir)
size = len(filenames)

for filename in filenames:
    filepath = meragefiledir + '\\' + filename
    for line in open(filepath):
        if i <= int(size * 0.6):
            file1.writelines(line.replace("\n", "") + '\t' + "1" + '\n')
        elif i <= int(size * 0.8):
            file2.writelines(line.replace("\n", "") + '\t' + "1" + '\n')
        else:
            file3.writelines(line.replace("\n", "") + '\t' + "1" + '\n')
        i += 1
file1.close()
file2.close()
file3.close()