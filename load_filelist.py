import torch.utils.data as data

import os
import os.path
import torch
import numpy as np

def default_list_reader(fileList):
    dataFileList = []
    labelList = []

    with open(fileList, 'r') as file:
        for line in file.readlines():
            line.strip('\n')
            line.rstrip()
            information = line.split()
            dataFileList.append(information[0])
            labelList.append([float(l) for l in information[1:len(information)]])

    return dataFileList, labelList

class FileListDataLoader(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=None, loader=None):
        self.root      = root
        self.transform = transform
        self.loader    = loader
        self.dataFileList = []
        self.labelList = []

        with open(fileList, 'r') as file:
            for line in file.readlines():
                line.strip('\n')
                line.rstrip()
                information = line.split()
                self.dataFileList.append(information[0])
                self.labelList.append([float(l) for l in information[1:len(information)]])

    def __getitem__(self, index):
        filePath = self.dataFileList[index]
        label = self.labelList[index]
        totalFilePath = os.path.join(self.root, filePath)
        numpy_array = np.load(totalFilePath)
        numpy_array = numpy_array / np.max(numpy_array) #最大值归一化
        # down_x, down_y = lttb.lttb(range(23399), numpy_array, 4000)
        #dataTensor = torch.FloatTensor(down_y)

        df_col_diff = np.diff(numpy_array, axis=0)

        dataTensor = torch.FloatTensor(numpy_array)#*256
        dataTensor_diff =  torch.FloatTensor(df_col_diff)
        #dataTensor = dataTensor*10
        if self.transform is not None:
            dataTensor = self.transform(dataTensor)
            dataTensor_diff = self.transform(dataTensor_diff)
        labelTensor = torch.FloatTensor(label)
        return dataTensor, dataTensor_diff, labelTensor

    def __len__(self):
        return len(self.dataFileList)