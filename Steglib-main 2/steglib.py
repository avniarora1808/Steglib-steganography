from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, ToFloat
from catalyst.dl import SupervisedRunner
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from glob import glob
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from skimage.io import imread
import torch.nn.functional as F
from scipy.special import softmax
from scipy.fftpack import fft, dct
import seaborn as sns
import re
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import jpegio as jio
from dotenv import dotenv_values
from PIL import Image

config = dotenv_values()
UPLOAD_FOLDER = config['UPLOAD_FOLDER']
OUTPUT_FOLDER = config['OUTPUT_FOLDER']

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_name('efficientnet-b0')
        self.dense_output = nn.Linear(1280, num_classes)
    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)

augment = Compose([
    ToFloat(max_value=255),
    ToTensorV2()
], p = 1)

class Alaska2Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        name = self.data.loc[idx][0]
        img = imread(name)
        img = augment(image = img)
        item = {'features': img['image']}
        return item

def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label

def lsb_encryption(filename, message, delim=""):
    img = Image.open(os.path.join(UPLOAD_FOLDER, filename))
    width, height = img.size
    array = np.array(list(img.getdata()))
    if img.mode == 'RGB':
        n = 3
    elif img.mode == 'RGBA':
        n = 4
    total_pixels = array.size // n
    message += delim
    b_message = ''.join([format(ord(i), "08b") for i in message])
    req_pixels = len(b_message)
    if req_pixels > total_pixels:
        print("ERROR: Need larger file size")
        return ""
    else:
        index = 0
        for p in range(total_pixels):
            for q in range(0, 3):
                if index < req_pixels:
                    array[p][q] = int(bin(array[p][q])[2:9] + b_message[index], 2)
                    index += 1
        array = array.reshape(height, width, n)
        enc_img = Image.fromarray(array.astype('uint8'), img.mode)
        enc_img.save(os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_encoded.png'))
        print("Image Encoded Successfully")
        return os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_encoded.png')

def lsb_decryption(filename, delim=None):
    img = Image.open(filename)
    array = np.array(list(img.getdata()))
    if img.mode == 'RGB':
        n = 3
    elif img.mode == 'RGBA':
        n = 4
    total_pixels = array.size // n
    hidden_bits = ""
    for p in range(total_pixels):
        for q in range(0, 3):
            hidden_bits += (bin(array[p][q])[2:][-1])
    hidden_bits = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]
    message = ""
    if not delim is None:
        for i in range(len(hidden_bits)):
            if message[len(delim):] == delim:
                break
            else:
                message += chr(int(hidden_bits[i], 2))
    else:
        for i in range(len(hidden_bits)):
            message += chr(int(hidden_bits[i], 2))
    if not delim is None and delim in message:
        return message[:len(delim)], True
    return message, False

def lsb_filter(path, n=2):
    print(path)
    image = Image.open(path)
    mask = (1 << n) - 1  # sets first bytes to 0
    color_data = [(255 * ((channel[0] & mask) + (channel[1] & mask) + (channel[2] & mask)) // (3 * mask),) * 3 for channel in image.getdata()]
    image.putdata(color_data)
    return image

def display_lsb(filename):
    enc_img = lsb_filter(os.path.join(UPLOAD_FOLDER, filename))
    enc_img.save(os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_lsb.png'))
    return os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_lsb.png')

def display_dct(filename):
    plt.clf()
    plt.figure(1, figsize = (10, 10))
    plt.title("dct_image")
    c_struct = jio.read(os.path.join(UPLOAD_FOLDER, filename))
    DCT = np.zeros([512, 512, 3])
    DCT[:, :, 0] = c_struct.coef_arrays[0]
    DCT[:, :, 1] = c_struct.coef_arrays[1]
    DCT[:, :, 2] = c_struct.coef_arrays[2]
    QTbl = c_struct.quant_tables[0]
    plt.imshow(abs(DCT))
    fig1 = plt.gcf()
    fig1.savefig(os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_dct.png'))
    #return filename.split('.')[0] + '_dct.png'
    return os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_dct.png')

def dct_histogram(filename):
    img = Image.open(os.path.join(UPLOAD_FOLDER, filename))
    np_image = np.array(img)
    np_image_size = np_image.shape
    np_dct_data = np.zeros(np_image_size)
    for i in np.r_[:np_image_size[0]:8]:
        for j in np.r_[:np_image_size[1]:8]:            
            np_dct_data[i:(i+8), j:(j+8)] = dct(np_image[i:(i+8),j:(j+8)])
    sample = dct(np_dct_data.flatten())
    plt.clf()
    plot = sns.kdeplot(sample, color="b")
    #fig2 = plot.get_figure()
    #fig2.savefig(os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_graph.png'))
    plot.get_figure().savefig(os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_graph.png'))
    return os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + '_graph.png')

def predict_possible(filename):
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices([filename])
        .map(decode_image)
        .batch(1)
    )
    model = keras.models.load_model('models/best_binary.h5')
    result = model.predict(test_dataset)
    return result[0][0] * 100

def predict_class(filename):
    class_names = ['NORMAL', 'JMiPOD_75', 'JMiPOD_90', 'JMiPOD_95', 
                'JUNIWARD_75', 'JUNIWARD_90', 'JUNIWARD_95',
                    'UERD_75', 'UERD_90', 'UERD_95']
    class_labels = { name: i for i, name in enumerate(class_names)}

    model = Net(num_classes=len(class_labels))
    model.load_state_dict(torch.load('models/best_multiclass.pth', map_location=torch.device('cpu')))

    runner = SupervisedRunner()
    filename = os.path.join(UPLOAD_FOLDER, filename)
    #test_filename = 'test.jpg'
    test_df = pd.DataFrame([filename], columns=['filename'])
    print(test_df)
    test_ds = Alaska2Dataset(test_df)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False)
    preds = []
    for outputs in tqdm(runner.predict_loader(loader=test_loader, model=model)):
        preds.append(softmax(outputs['logits']))
    preds = np.array(preds)
    #print(preds[0][0])
    df_final = pd.DataFrame(columns=class_names)
    df_final = df_final.append(pd.DataFrame(preds[0][0].reshape(1, -1), columns = list(df_final)))
    jmipod_sum = ['JMiPOD_75', 'JMiPOD_90', 'JMiPOD_95']
    juniward_sum = ['JUNIWARD_75', 'JUNIWARD_90', 'JUNIWARD_95']
    uerd_sum = ['UERD_75', 'UERD_90', 'UERD_95']
    df_final['JMiPOD'] = df_final[jmipod_sum].sum(axis = 1)
    df_final['JUNIWARD'] = df_final[juniward_sum].sum(axis = 1)
    df_final['UERD'] = df_final[uerd_sum].sum(axis = 1)
    print(df_final)
    class_names.remove('NORMAL')
    encoding_type = df_final[['JMiPOD', 'JUNIWARD', 'UERD']].idxmax(axis=1)
    specific_type = df_final[class_names].idxmax(axis = 1)
    return encoding_type.iloc[0], specific_type.iloc[0]