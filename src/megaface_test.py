import os 
import argparse
import numpy as np
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import onnxruntime  
import tritonclient.http as httpclient

from src.dino.utils import make_square 

def get_parse():
    parser = argparse.ArgumentParser(description='megaface test')
    parser.add_argument('--root_dir', default='/mnt/data/megaface_clean')
    parser.add_argument('--face_folders', default='faces,facescrub_images')
    parser.add_argument('--mega_folder', default='megaface_images')
    parser.add_argument('--resize', type=str, default='112,112')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-n', '--model_name', default='onnx')
    parser.add_argument('-p', '--model_path', default='backbone.onnx')
    return parser.parse_args()

def build_model(model_name, model_path):
    if model_name == 'onnx':
        model = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    elif model_name[:3] == 'trt':
        triton_host = '127.0.0.1:8000'
        triton_client = httpclient.InferenceServerClient(triton_host)
        model = triton_client
    else:
        raise ValueError(model_name)
    return model

def model_forward(model_name, model, inputs):
    if model_name == 'onnx':
        inputs_numpy = inputs.detach().numpy()
        features = model.run(['features_norm'], {'input': inputs_numpy})[0]
    elif model_name[:3] == 'trt':
        inputs_numpy = inputs.detach().numpy()
        trt_inputs = [httpclient.InferInput('input', inputs_numpy.shape, 'FP32')]
        trt_inputs[0].set_data_from_numpy(inputs_numpy)
        trt_outputs = [httpclient.InferRequestedOutput('features_norm')]
        response = model.infer(model_name[4:], trt_inputs, request_id=str(1), outputs=trt_outputs)
        features = response.as_numpy('features_norm')
        return features
    else:
        inputs_cuda = inputs.to('cuda')
        features = model(inputs_cuda)[1].to('cpu').detach().numpy()
    return features

def preprocess(img_bgr):
    img_input = ((img_bgr / 255.) - 0.5) / 0.5
    tensor = torch.tensor(img_input.transpose(2, 0, 1)).float()
    return tensor

class MegaDataset(Dataset):
    def __init__(self, path_list, resize_hw, make_square=False):
        self.images = []
        self.labels = []
        self.resize = (resize_hw[1], resize_hw[0])
        self.make_square = make_square

        for line in path_list:
            line = line.strip()
            self.images.append(line)
            self.labels.append(line)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.images[idx])
        img_bgr = cv2.resize(img_bgr, self.resize)
        if self.make_square: img_bgr = make_square(img_bgr)
        img_input = preprocess(img_bgr)

        label = self.labels[idx]

        return img_input, label

def get_mega_dataloader(root_dir, folder, batch_size, resize_hw=(112, 112), make_square=False):
    path_list = []
    for name in os.listdir(os.path.join(root_dir, folder)):
        for img_name in os.listdir(os.path.join(root_dir, folder, name)):
            img_path = os.path.join(root_dir, folder, name, img_name)
            if img_name[-3:] in ['jpg', 'png']:
                path_list.append(img_path)
            elif '.' not in img_name:
                for img_name2 in os.listdir(img_path):
                    path_list.append(os.path.join(img_path, img_name2))
    dataset = MegaDataset(path_list, resize_hw, make_square)
    dataloader = DataLoader(dataset, batch_size, num_workers=8)
    return dataloader

@torch.no_grad()
def get_acc(model, model_name, face_dataloaders, mega_dataloader, print_info=True):
    face_datas = [{} for _ in range(len(face_dataloaders))] 
    mega_data = {}

    if print_info: print('extract face features ------')
    for idx in range(len(face_dataloaders)):
        for inputs, labels in tqdm(face_dataloaders[idx]):
            features = model_forward(model_name, model, inputs)

            for feature, label in zip(features, labels):
                face_datas[idx][label] = feature

    if print_info: print('extract mega features ------')
    for inputs, labels in tqdm(mega_dataloader):
        features = model_forward(model_name, model, inputs)

        for feature, label in zip(features, labels):
            mega_data[label] = feature

    mega_arr = []
    for key, val in mega_data.items():
        mega_arr.append(val)
    mega_arr = np.array(mega_arr)
    
    if print_info: print(f'megaface arr: {mega_arr.shape} ------')

    acc_arr = []
    for idx in range(len(face_dataloaders)):
        face_data = face_datas[idx]
        face_dict = {}
        for key, val in face_data.items():
            k = key.split('/')[-2]

            arr = face_dict.get(k, [])
            arr.append(val)
            face_dict[k] = arr 
        if print_info: print({key: len(val) for key, val in face_dict.items()})

        count = [0, 0]

        with tqdm(face_dict.items()) as t:
            for key, val in t:
                arr = np.array(val)
                arr_dot = np.dot(arr, arr.T)

                matrix = np.dot(arr, mega_arr.T)
                matrix_max = [max(m) for m in matrix]

                for i in range(len(arr)):
                    for j in range(i + 1, len(arr), 1):
                        if arr_dot[i][j] > matrix_max[i]: count[0] += 1
                        if arr_dot[i][j] > matrix_max[j]: count[0] += 1
                        count[1] += 2

                t.set_postfix(str=f'acc: {count[0]/count[1]*100:.2f}%')

        if print_info: print(f'acc: {count[0]/count[1]*100:.2f}%, {count[0]}/{count[1]} ------')
        acc_arr.append(count[0]/count[1])

    return acc_arr

if __name__ == '__main__':
    args = get_parse()

    model = build_model(args.model_name, args.model_path)
    print('model build done ------')

    resize_hw = args.resize.split(',')
    resize_hw = [int(size) for size in resize_hw]
    face_folders = args.face_folders.split(',')
    face_dataloaders = [get_mega_dataloader(args.root_dir, folder, args.batch_size, resize_hw) for folder in face_folders]
    mega_dataloader = get_mega_dataloader(args.root_dir, args.mega_folder, args.batch_size, resize_hw)

    acc_arr = get_acc(model, args.model_name, face_dataloaders, mega_dataloader)
    print('acc arr:', acc_arr)