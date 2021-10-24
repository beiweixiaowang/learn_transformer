# 使用transformer实现OCR字符识别 
- **Writer** : 永驻
- **Date** : 2021.10.24

本次任务以ICDAR2015 Incidental Scene Text中的[Task4.3:word recognition](https://rrc.cvc.uab.es/?ch=4&com=downloads)单词识别为子任务作为数据集。讲解如何**使用transformer来实现一个简单的OCR文字识别任务**，并从中体会transformer是如何应用到除分类以外更复杂的CV任务中的。

文章从以下几个方面进行讲解：
- 数据集简介
- 数据集分析与字符映射关系构建
- 如何将transformer引入OCR
- 构建训练框架

包含以下几文件：
- my_transformer.py （上文中构建的transformer）
- analysis_recognition_dataset.py（数据分析）
- train_utils.py（训练辅助函数）
- ocr_by_transformer.py (OCR任务训练脚本)

下面进入正式内容：
***
## 数据集简介
本文OCR实验使用的数据集基于`ICDAR2015 Incidental Scene Text` 中的 `Task 4.3: Word Recognition`，这是一个简单的单词识别任务。
数据集地址：[下载链接](https://pan.baidu.com/s/1TIvYgkn_Q5Z9Nl0amwGZzg)
提取码：qa8v
该数据集包含了众多自然场景图像中出现的文字区域，原始数据中训练集含有4468张图像，测试集含有2077张图像，他们都是从原始大图中依据文字区域的bounding box裁剪出来的，图像中的文字基本处于图片中心位置。

数据集中的图像类似如下样式：
|word_34.png "SHOP"  |word_241.png "Line" |
| ------ | ------ |
|  ![34](./img/word_34.png) | ![241](./img/word_241.png) | 

数据集结构大致如下
- train
- train_gt.txt
- valid
- valid.txt

为了简化后续实验的识别难度，提供的数据集**使用高宽比>1.5粗略过滤了文字竖向排列的图像**，因此与ICDAR2015的原始数据集略有差别。

## 数据分析与字符映射关系构建
构建analysis_recognition_dataset.py脚本来对数据进行简单分析
这个脚本的作用是：
- 对数据进行标签字符统计
- 最长标签长度统计
- 图像尺寸分析
- 构建字符标签的映射关系文件`lbl2id_map.txt`

首先进行准备工作，导入需要的库，设置相关路径
```python
import os  
import cv2
base_data_dir = '../ICDAR_2015/'

train_img_dir = os.path.join(base_data_dir, 'train')
valid_img_dir = os.path.join(base_data_dir, 'valid')
train_lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
valid_lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')
lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
```

### 1、统计标签文件中都包含哪些label以及各自出现的次数
```python
def statistics_label_cnt(lbl_path, lbl_cnt_map):
    """
    统计标签文件中都包含哪些label以及各自出现的次数
    """
    with open(lbl_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            items = line.rstrip().split(',')
            img_name = items[0]
            lbl_str = items[1].strip()[1:-1]
            for lbl in lbl_str:
                if lbl not in lbl_cnt_map.keys():
                    lbl_cnt_map[lbl] = 1
                else:
                    lbl_cnt_map[lbl] += 1
```

### 2、统计标签文件中最长的label所包含的字符数
```python
def statistics_max_len_label(lbl_path):
    """
    统计标签文件中最长的label所包含的字符数
    """
    max_len = -1
    with open(lbl_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            items = line.rstrip().split(',')
            img_name = items[0]
            lbl_str = items[1].strip()[1:-1]
            lbl_len = len(lbl_str)
            max_len = max_len if max_len > lbl_len else lbl_len
    return max_len
```
具体效果如下：
```
数据集中包含字符最多的label长度为21
训练集中出现的label
{'[': 2, '0': 182, '6': 38, ']': 2, '2': 119, '-': 68, '3': 50, 'C': 593, 'a': 843, 'r': 655, 'p': 197, 'k': 96, 'E': 1421, 'X': 110, 'I': 861, 'T': 896, 'R': 836, 'f': 133, 'u': 293, 's': 557, 'i': 651, 'o': 659, 'n': 605, 'l': 408, 'e': 1055, 'v': 123, 'A': 1189, 'U': 319, 'O': 965, 'N': 785, 'c': 318, 't': 563, 'm': 202, 'W': 179, 'H': 391, 'Y': 229, 'P': 389, 'F': 259, 'G': 345, '?': 5, 'S': 1161, 'b': 88, 'h': 299, ' ': 50, 'g': 171, 'L': 745, 'M': 367, 'D': 383, 'd': 257, '$': 46, '5': 77, '4': 44, '.': 95, 'w': 97, 'B': 331, '1': 184, '7': 43, '8': 44, 'V': 158, 'y': 161, 'K': 163, '!': 51, '9': 66, 'z': 12, ';': 3, '#': 16, 'j': 15, "'": 51, 'J': 72, ':': 19, 'x': 27, '%': 28, '/': 24, 'q': 3, 'Q': 19, '(': 6, ')': 5, '\\': 8, '"': 8, '´': 3, 'Z': 29, '&': 9, 'É': 1, '@': 4, '=': 1, '+': 1}
训练集+验证集中出现的label
{'[': 2, '0': 232, '6': 44, ']': 2, '2': 139, '-': 87, '3': 69, 'C': 893, 'a': 1200, 'r': 935, 'p': 317, 'k': 137, 'E': 2213, 'X': 181, 'I': 1241, 'T': 1315, 'R': 1262, 'f': 203, 'u': 415, 's': 793, 'i': 924, 'o': 954, 'n': 880, 'l': 555, 'e': 1534, 'v': 169, 'A': 1827, 'U': 467, 'O': 1440, 'N': 1158, 'c': 442, 't': 829, 'm': 278, 'W': 288, 'H': 593, 'Y': 341, 'P': 582, 'F': 402, 'G': 521, '?': 7, 'S': 1748, 'b': 129, 'h': 417, ' ': 82, 'g': 260, 'L': 1120, 'M': 536, 'D': 548, 'd': 367, '$': 57, '5': 100, '4': 53, '.': 132, 'w': 136, 'B': 468, '1': 228, '7': 60, '8': 51, 'V': 224, 'y': 231, 'K': 253, '!': 65, '9': 76, 'z': 14, ';': 3, '#': 24, 'j': 19, "'": 70, 'J': 100, ':': 24, 'x': 38, '%': 42, '/': 29, 'q': 3, 'Q': 28, '(': 7, ')': 5, '\\': 8, '"': 8, '´': 3, 'Z': 36, '&': 15, 'É': 2, '@': 9, '=': 1, '+': 2, 'é': 1}

```
上方代码中，lbl_cnt_map 为字符出现次数的统计字典，后面还会用于建立字符及其id映射关系。

从数据集统计结果来看，测试集含有训练集没有出现过的字符，例如测试集中包含1个'©'未曾在训练集出现。这种情况数量不多，应该问题不大，所以此处未对数据集进行额外处理(但是有意识的进行这种训练集和测试集是否存在diff的检查是必要的)。



### 3、char和id的映射字典构建
在本文OCR任务中，需要对图片中的每个字符进行预测，为了达到这个目的，首先就需要建立一个字符与其id的映射关系，将文本信息转化为可供模型读取的数字信息，这一步类似NLP中建立语料库。

在构建映射关系时，除了记录所有标签文件中出现的字符外，还需要初始化三个特殊字符，分别用来代表一个 **句子起始符、句子终止符和填充(Padding)** 标识符。相信经过6.1节的介绍你能够明白这3种特殊字符的作用，后面dataset构建部分的讲解也还会再次提到。

脚本运行后，所有字符的映射关系将会保存在 lbl2id_map.txt文件中。

构建映射代码：
```python
 # 构造 label - id 之间的映射
    print("\n\n构造 label - id 之间的映射")
    lbl2id_map = dict()
    # 初始化两个特殊字符
    lbl2id_map['🤍'] = 0    # padding标识符
    lbl2id_map['■'] = 1    # 句子起始符
    lbl2id_map['□'] = 2    # 句子结束符
    # 生成其余label的id映射关系
    cur_id = 3
    for lbl in lbl_cnt_map.keys():
        lbl2id_map[lbl] = cur_id
        cur_id += 1
    # 保存 label - id 之间的映射
    with open(lbl2id_map_path, 'w', encoding='utf-8') as writer:
        for lbl in lbl2id_map.keys():
            cur_id = lbl2id_map[lbl]
            print(lbl, cur_id)
            line = lbl + '\t' + str(cur_id) + '\n'
            writer.write(line)
```
构建出来的映射如下：
```
🤍 0
■ 1
□ 2
[ 3
0 4
6 5
] 6
2 7
- 8
3 9
C 10
```

读取label-id映射关系记录文件
```python
def load_lbl2id_map(lbl2id_map_path):
    """
    读取label-id映射关系记录文件
    """
    lbl2id_map = dict()
    id2lbl_map = dict()
    with open(lbl2id_map_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            items = line.rstrip().split('\t')
            label = items[0]
            cur_id = int(items[1])
            lbl2id_map[label] = cur_id
            id2lbl_map[cur_id] = label
    return lbl2id_map, id2lbl_map
```
### 4、图像尺寸的分析
```python
print("\n\n 分析数据集图片尺寸")
    min_h = 1e10
    min_w = 1e10
    max_h = -1
    max_w = -1
    min_ratio = 1e10
    max_ratio = 0
    for img_name in os.listdir(train_img_dir):
        img_path = os.path.join(train_img_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        ratio = w / h
        min_h = min_h if min_h <= h else h
        max_h = max_h if max_h >= h else h
        min_w = min_w if min_w <= w else w
        max_w = max_w if max_w >= w else w
        min_ratio = min_ratio if min_ratio <= ratio else ratio
        max_ratio = max_ratio if max_ratio >= ratio else ratio
    print("min_h", min_h)
    print("max_h", max_h)
    print("min_w", min_w)
    print("max_w", max_w)
    print("min_ratio", min_ratio)
    print("max_ratio", max_ratio)
```
结果如下：
```
分析数据集图片尺寸
min_h 9
max_h 295
min_w 16
max_w 628
min_ratio 0.6666666666666666
max_ratio 8.619047619047619
```
## 将transformer引入OCR
transformer广泛应用于NLP领域中，可以解决类似机器翻译这种`sequence to sequence`问题，如下图
![sts](./img/sts.jpg)
对于ocr问题，我们希望把![241](./img/data_share.png)识别为“Share”,可以认为是一个 `image to sequence`问题。**如果让image变为sequence， 那么ocr任务也就变成了一个sequence to sequence问题，使用transformer解决也就合理了。** 剩下的问题就是如何将图片信息构造成transformer想要的，类似于 word embedding 形式的输入。

回到我们的任务，既然待预测的图片都是长条状的，文字基本都是水平排列，那么我们将特征图沿水平方向进行整合，得到的每一个embedding可以认为是图片纵向的某个切片的特征，将这样的特征序列交给transformer，利用其强大的attention能力来完成预测。

因此，基于以上分析，我们将模型框架的pipeline定义为下图所示的形式：
![ocr](./img/ocr_by_transformer.png)

通过观察上图可以发现，整个pipeline和利用transformer训练机器翻译的流程是基本一致的，之间的差异主要是多了 **借助一个CNN网络作为backbone提取图像特征得到input embedding的过程。**

关于构造transformer的输入embedding这部分的设计，是本文的重点，也是整个算法能够work的关键。后文会结合代码，对上面示意图中展示的相关细节进行展开讲解.

## 训练框架代码详解

训练框架相关代码实现在 ocr_by_transformer.py 文件中
下面开始逐步讲解代码，主要有以下几个部分：
- 构建dataset → 图像预处理、label处理等
- 模型构建 → backbone + transformer
- 模型训练
- 推理 → 贪心解码

### 1、准备工作
导入库并设置基础参数
```python

import os
import time
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from analysis_recognition_dataset import load_lbl2id_map, statistics_max_len_label
from my_transformer import *
from train_utils import *

import warnings
warnings.filterwarnings("ignore")

base_data_dir = '../ICDAR_2015'
device = torch.device("cuda")
nrof_epochs = 100
batch_size = 64 #这里笔者GPU为两块3090，如果设备显存不够可以适当调小
model_save_path = "../model_path/orc_model.pth"
```
读取图像label中的字符与其id的映射字典
```python
lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)

train_lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
valid_lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')
train_max_label_len = statistics_max_len_label(train_lbl_path)
valid_max_label_len = statistics_max_len_label(valid_lbl_path)
sequence_len = max(train_max_label_len, valid_max_label_len)
```
### 2.Dataset构建
#### 2.1 图片预处理方案
假设图片尺寸为[batch_size, $3, H_i, W_i$]
经过网络后的特征图尺寸为[batch_size, $C_f, H_f, W_f$]
基于之前对于数据集的分析，图片基本都是水平长条状的，图像内容是水平排列的字符组成的单词。那么图片空间上同一纵向切片的位置，基本只有一个字符，因此纵向分辨率不需要很大，那么取$H_f = 1$。而横向分辨率需要大一些，我们需要不同embedding来编码水平方向上不同字符的特征。
![adf](./img/img2feature.jpg)
我们采用最经典的**resnet18网络来作为backbone,** 由于其下采样倍数为32，最后一层特征图channel数为512，那么:
$H_i = H_f * 32 = 32$
$C_f = 512$
有两种方案来确定输入图片的宽度：
![a](./img/two_resize.jpg)
- **方法一：** 设定一个固定尺寸，将图像保持其宽高比进行resize，右侧空余区域进行padding
- **方法二：** 直接将原始图像强制resize到一个预设的固定尺寸

这里选择方法一，因为图片的宽高比和图片中单词的字符数量是大致呈正比的，如果预处理时保持住原图片的宽高比，那么特征图上每一个像素对应原图上字符区域的范围就是基本稳定的，这样或许有更好的预测效果。
这里还有个细节，观察上图你会发现，每个宽：高=1:1的区域内，基本都分布着2-3个字符，因此我们实际操作时也没有严格的保持宽高比不变，而是将宽高比提升了3倍，即先将原始图片宽度拉长到原来的3倍，再保持宽高比，将高resize到32。
>这样做的目的是让图片上每一个字符，都有至少一个特征图上的像素与之对应，而不是特征图宽维度上一个像素，同时编码了原图中的多个字符的信息，这样我认为会对transformer的预测带来不必要的困难

确定了resize方案，$Wi$具体设置为多少呢？结合前面我们对数据集做分析时的两个重要指标，数据集label中最长字符数为21，最长的宽高比8.6，我们将最终的宽高比设置为 24:1，因此汇总一下各个参数的设置：
$H_i = H_f * 32 = 32$
$W_i = 24 * H_i = 768$
$C_f = 512, H_f = 1, W_f = 24$

代码实现：
```python
w, h = img.size
ration = round((w / h) * 3)
if ration == 0:
    ration = 1
if ration > self.max_ration:
    ration = self.max_ration
h_new = 32
w_new = h_new * ration
img_resize = img.resize((w_new, h_new), Image.BILINEAR)

img_padd = Image.new("RGB", (32*self.max_ration, 32), (0, 0, 0))
img_padd.paste(img_resize, (0, 0))
```
完整代码：
```python
class Recognition_Dataset(object):
    def __init__(self, dataset_root_dir, lbl2id_map, sequence_len, max_ration, phase="train", pad=0):

        if phase == 'train':
            self.img_dir = os.path.join(base_data_dir, 'train')
            self.lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
        else:
            self.img_dir = os.path.join(base_data_dir, 'valid')
            self.lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')

        self.lbl2id_map = lbl2id_map
        self.pad = pad
        self.sequence_len = sequence_len
        self.max_ration = max_ration * 3
        self.imgs_list = []
        self.lbls_list = []

        with open(self.lbl_path, 'r', encoding="utf-8") as reader:
            for line in reader:
                items = line.rstrip().split(',')
                img_name = items[0]
                lbl_str = items[1].strip()[1:-1]

                self.imgs_list.append(img_name)
                self.lbls_list.append(lbl_str)

        self.color_trans = transforms.ColorJitter(0.1, 0.1, 0.1)
        self.trans_Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.457, 0.456], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_str = self.lbls_list[index]

        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        ration = round((w / h) * 3)
        if ration == 0:
            ration = 1
        if ration > self.max_ration:
            ration = self.max_ration
        h_new = 32
        w_new = h_new * ration
        img_resize = img.resize((w_new, h_new), Image.BILINEAR)

        img_padd = Image.new("RGB", (32*self.max_ration, 32), (0, 0, 0))
        img_padd.paste(img_resize, (0, 0))

        img_input = self.color_trans(img_padd)
        img_input = self.trans_Normalize(img_input)


        encode_mask = [1]*ration + [0] * (self.max_ration - ration)
        encode_mask = torch.tensor(encode_mask)
        encode_mask = (encode_mask != 0).unsqueeze(0)
        gt = []
        gt.append(1)
        for lbl in lbl_str:
            gt.append(self.lbl2id_map[lbl])
        gt.append(2)
        for i in range(len(lbl_str), self.sequence_len):
            gt.append(0)
        gt = gt[:self.sequence_len]

        decode_in = gt[:-1]
        decode_in = torch.tensor(decode_in)
        decode_out = gt[1:]
        decode_out = torch.tensor(decode_out)
        decode_mask = self.make_std_mask(decode_in, self.pad)
        ntokens = (decode_out != self.pad).data.sum()

        return img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.squeeze(0)
        return tgt_mask

    def __len__(self):
        return len(self.imgs_list)
```
上面的代码还有几个和label处理相关的细节，属于Transformer训练相关的逻辑。

**encode_encode_mask**
由于我们对图像进行了尺寸调整，并根据需求对图像进行了padding，而padding的位置是没有包含有效信息的，为此需要根据padding比例构造相应encode_mask，让transformer在计算时忽略这部分无意义的区域。
```python
encode_mask = [1]*ration + [0] * (self.max_ration - ration)
encode_mask = torch.tensor(encode_mask)
encode_mask = (encode_mask != 0).unsqueeze(0)
```
**label处理**
由于我们对图像进行了尺寸调整，并根据需求对图像进行了padding，而padding的位置是没有包含有效信息的，为此需要根据padding比例构造相应encode_mask，让transformer在计算时忽略这部分无意义的区域。

**decode_mask**
一般的在decoder中我们会根据label的sequence_len生成一个上三角阵形式的mask，mask的每一行便可以控制当前time_step时，只允许decoder获取当前步时之前的字符信息，而禁止获取未来时刻的字符信息，这防止了模型训练时的作弊行为。

decode_mask经过一个特殊的函数 make_std_mask() 进行生成。

同时，decoder的label制作同样要考虑上对padding的部分进行mask，所以decode_mask在label被padding对应的位置处也应该进行写成False。
![mask](./img/decode_mask.png)
```py
 def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.squeeze(0)
        return tgt_mask
```
以上是构建Dataset的所有细节，进而我们可以构建出DataLoader供训练使用
```py
max_ratio = 8
train_dataset = Recognition_Dataset(base_data_dir, lbl2id_map, sequence_len, max_ratio, 'train', pad=0)
valid_dataset = Recognition_Dataset(base_data_dir, lbl2id_map, sequence_len, max_ratio, 'valid', pad=0)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4)

```
###3.模型构建
代码通过 make_ocr_model 和 OCR_EncoderDecoder 类完成模型结构搭建。

可以从 make_ocr_model 这个函数看起，该函数首先**调用了pytorch中预训练的Resnet-18作为backbone以提取图像特征，** 此处也可以根据自己需要调整为其他的网络，但需要**重点关注的是网络的下采样倍数，以及最后一层特征图的channel_num，相关模块的参数需要同步调整。** 之后调用了 OCR_EncoderDecoder 类完成transformer的搭建。最后对模型参数进行初始化。

在 OCR_EncoderDecoder 类中，该类相当于是一个transformer各基础组件的拼装线，包括 encoder 和 decoder 等，其初始参数是已存在的基本组件，其基本组件代码都在my_transformer.py文件中，本文将不在过多叙述。

这里再来回顾一下，图片经过backbone后，如何构造为Transformer的输入：

图片经过backbone后将输出一个维度为 [batch_size, 512, 1, 24] 的特征图，在不关注batch_size的前提下，每一张图像都会得到如下所示具有512个通道的1×24的特征图，如图中红色框标注所示，将不同通道相同位置的特征值拼接组成一个新的向量，并作为一个时间步的输入，此时变构造出了维度为 [batch_size, 24, 512] 的输入，满足Transformer的输入要求。
![trsn](./img/transpose.jpg)
完整代码：
```py
class OCR_EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, src_position, tgt_embed, generator):
        super(OCR_EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_position = src_position
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        res = self.decode(memory, src_mask, tgt, tgt_mask)
        return res

    def encode(self, src, src_mask):
        src_embedds = self.src_embed(src)
        src_embedds = src_embedds.squeeze(-2)
        src_embedds = src_embedds.permute(0, 2, 1)

        src_embedds = self.src_position(src_embedds)

        return self.encoder(src_embedds, src_mask)


    def decode(self, memory, src_mask, tgt, tgt_mask):
        target_embedds = self.tgt_embed(tgt)
        return self.decoder(target_embedds, memory, src_mask, tgt_mask)



def make_ocr_model(tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    backbone = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])

    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = OCR_EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        backbone,
        c(position),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    for child in model.children():
        if child is backbone:
            for param in child.parameters():
                param.requires_grad = False
            continue
        for p in child.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model
```
通过上面的类和函数，可以方便的构建transformer模型：
```py
    tgt_vocab = len(lbl2id_map.keys())
    d_model = 512
    ocr_model = make_ocr_model(tgt_vocab, N=5, d_model=d_model, d_ff=2048, h=8, dropout=0.1)
    ocr_model.to(device)
```
### 4.模型训练
模型训练之前，还需要定义**模型评判准则、** **迭代优化器**等。本实验在训练时，使用了**标签平滑（label smoothing）、网络训练热身（warmup）** 等策略，以上策略的调用函数均在train_utils.py文件中，此处不涉及以上两种方法的原理及代码实现。

label smoothing可以将原始的硬标签转化为软标签，从而增加模型的容错率，提升模型泛化能力。代码中 LabelSmoothing() 函数实现了label smoothing，同时内部使用了相对熵函数计算了预测值与真实值之间的损失。
```py
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
```

warmup策略能够有效控制模型训练过程中的优化器学习率，自动化的实现模型学习率由小增大再逐渐下降的控制，帮助模型在训练时更加稳定，实现损失的快速收敛。代码中 NoamOpt() 函数实现了warmup控制，采用的Adam优化器，实现学习率随迭代次数的自动调整。
```py
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
```
进行上述操作：
```py
criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.0)  
optimizer = torch.optim.Adam(ocr_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
model_opt = NoamOpt(d_model, 1, 400, optimizer)
```
**SimpleLossCompute()** 类实现了transformer输出结果的loss计算。在使用该类直接计算时，类需要接收(x, y, norm)三个参数，x为decoder输出的结果，y为标签数据，norm为loss的归一化系数，用batch中所有有效token数即可。由此可见，此处才正完成transformer所有网络的构建，实现数据计算流的流通。
```py
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):

        x = self.generator(x)
        x_ = x.contiguous().view(-1, x.size(-1))
        y_ = y.contiguous().view(-1)
        loss = self.criterion(x_, y_)
        loss /= norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.item() * norm
```
**run_epoch()** 函数内部完成了一个epoch训练的所有工作，包括数据加载、模型推理、损失计算与方向传播，同时将训练过程信息进行打印。
```py
def run_epoch(data_loader, model, loss_compute, device=None):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)
        decode_in = decode_in.to(device)
        decode_out = decode_out.to(device)
        decode_mask = decode_mask.to(device)
        ntokens = torch.sum(ntokens).to(device)

        out = model.forward(img_input, decode_in, encode_mask, decode_mask)

        loss = loss_compute(out, decode_out, ntokens)
        total_loss += loss
        total_tokens += tokens
        tokens += ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
```
### 5. 贪心解码
我们使用最简单的贪心解码直接进行OCR结果预测。因为模型每一次只会产生一个输出，我们选择输出的概率分布中的最高概率对应的字符为本次预测的结果，然后预测下一个字符，这就是所谓的贪心解码，见代码中 greedy_decode() 函数。

实验中分别将每一张图像作为模型的输入，逐张进行贪心解码统计正确率，并最终给出了训练集和验证集各自的预测准确率。

**greedy_decode()** 函数实现。
```py
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        next_word = torch.ones(1, 1).type_as(src.data).fill_(next_word).long()
        ys = torch.cat([ys, next_word], dim=1)

        next_word = int(next_word)
        if next_word == end_symbol:
            break
    ys = ys[0, 1:]
    return ys
```
结果如下：
```py
greedy decode trainset
----
tensor([43, 34, 11, 51, 23, 21, 35,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0])
tensor([43, 24, 21, 25, 51,  2])
----
tensor([17, 32, 18, 19, 31, 50, 30, 10, 30, 10, 17, 32, 41, 55, 55, 55,  2,  0,
         0,  0])
tensor([17, 32, 18, 19, 31, 50, 30, 10, 30, 10, 17, 32, 41, 55, 55, 55, 55, 55,
        55, 55])
----
tensor([17, 32, 18, 19, 31, 50, 30, 10, 17, 32, 41, 55, 55,  2,  0,  0,  0,  0,
         0,  0])
tensor([17, 32, 18, 19, 31, 50, 30, 10, 17, 32, 41, 55, 55, 55,  2])
----
tensor([49, 49, 29,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0])
tensor([49, 49, 49, 29,  2])
----
tensor([78, 46, 88,  5, 53, 79, 46,  5, 59,  9,  7, 46,  7, 65,  4,  4,  2,  0,
         0,  0])
tensor([78, 46, 88,  5, 53, 79, 46,  5, 59,  9,  7, 46,  7, 65,  4,  2])
total correct rate of trainset: 99.88441978733242%
```
最后将预测出来的张量进行映射就可以达到识别出的字符了。

## last word
我们使用ICDAR2015中的一个单词识别任务数据集，然后对数据的特点进行了简单分析，并构建了识别用的字符映射关系表。之后，我们重点介绍了将transformer引入来解决OCR任务的动机与思路，并结合代码详细介绍了细节，最后我们大致过了一些训练相关的逻辑和代码。

大致介绍transformer在CV领域的应用方法，希望帮助大家打开思路。❤
![](./img/hf.svg)