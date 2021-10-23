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
            # 将backbone的权重设为不计算梯度
            for param in child.parameters():
                param.requires_grad = False
            # 预训练好的backbone不进行随机初始化，其余模块进行随机初始化
            continue
        for p in child.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model

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

def judge_is_correct(pred, label):
    # 判断模型预测结果和label是否一致
    pred_len = pred.shape[0]
    label = label[:pred_len]
    is_correct = 1 if label.equal(pred) else 0
    return is_correct

if __name__ == '__main__':
    base_data_dir = '../ICDAR_2015'
    device = torch.device("cuda")

    nrof_epochs = 1500
    batch_size = 16

    model_save_path = "../model_path/orc_model.pth"

    lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
    lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)

    train_lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
    valid_lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')
    train_max_label_len = statistics_max_len_label(train_lbl_path)
    valid_max_label_len = statistics_max_len_label(valid_lbl_path)
    sequence_len = max(train_max_label_len, valid_max_label_len)

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

    tgt_vocab = len(lbl2id_map.keys())
    d_model = 512
    ocr_model = make_ocr_model(tgt_vocab, N=5, d_model=d_model, d_ff=2048, h=8, dropout=0.1)
    ocr_model.to(device)

    criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.0)
    optimizer = torch.optim.Adam(ocr_model.parameters(), lr=0, betas=(0.9, 0.98), eps= 1e-9)
    model_opt = NoamOpt(d_model, 1, 400, optimizer)

    for epoch in range(nrof_epochs):
        print(f"\nepoch{epoch}")
        print("train...")
        ocr_model.train()

        loss_compute = SimpleLossCompute(ocr_model.generator, criterion, model_opt)
        train_mean_loss = run_epoch(train_loader, ocr_model, loss_compute, device)

        if epoch % 10 == 0:
            print("valid...")
            ocr_model.eval()
            valid_loss_compute = SimpleLossCompute(ocr_model.generator, criterion, None)
            valid_mean_loss = run_epoch(valid_loader, ocr_model, valid_loss_compute, device)
            print(f"valid loss:{valid_mean_loss}")
    torch.save(ocr_model.state_dict(), model_save_path)
    ocr_model.eval()
    print("\n------------------------------------------------")
    print("greedy decode trainset")
    total_img_num = 0
    total_correct_num = 0
    for batch_idx, batch in enumerate(train_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)

        bs = img_input.shape[0]
        for i in range(bs):
            cur_img_input = img_input[i].unsqueeze(0)
            cur_encode_mask = encode_mask[i].unsqueeze(0)
            cur_decode_out = decode_out[i]

            pred_result = greedy_decode(ocr_model, cur_img_input, cur_encode_mask, max_len=sequence_len, start_symbol=1,
                                        end_symbol=2)
            pred_result = pred_result.cpu()

            is_correct = judge_is_correct(pred_result, cur_decode_out)
            total_correct_num += is_correct
            total_img_num += 1
            if not is_correct:
                # 预测错误的case进行打印
                print("----")
                print(cur_decode_out)
                print(pred_result)
    total_correct_rate = total_correct_num / total_img_num * 100
    print(f"total correct rate of trainset: {total_correct_rate}%")

    print("\n------------------------------------------------")
    print("greedy decode validset")
    total_img_num = 0
    total_correct_num = 0
    for batch_idx, batch in enumerate(valid_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)

        bs = img_input.shape[0]
        for i in range(bs):
            cur_img_input = img_input[i].unsqueeze(0)
            cur_encode_mask = encode_mask[i].unsqueeze(0)
            cur_decode_out = decode_out[i]

            pred_result = greedy_decode(ocr_model, cur_img_input, cur_encode_mask, max_len=sequence_len, start_symbol=1,
                                        end_symbol=2)
            pred_result = pred_result.cpu()

            is_correct = judge_is_correct(pred_result, cur_decode_out)
            total_correct_num += is_correct
            total_img_num += 1
            if not is_correct:
                # 预测错误的case进行打印
                print("----")
                print(cur_decode_out)
                print(pred_result)
    total_correct_rate = total_correct_num / total_img_num * 100
    print(f"total correct rate of validset: {total_correct_rate}%")