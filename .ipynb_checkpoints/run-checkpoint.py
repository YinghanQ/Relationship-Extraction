#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from config import Config
from utils import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from model.att_blstm import Att_BLSTM
from model.blstm import BLSTM
from model.blstm_cnn import BLSTM_CNN
from model.multi_att_blstm import Multi_Att_BLSTM
from model.att_blstm_cnn import Att_BLSTM_CNN
from evaluate import Eval


def print_result(predict_label, id2rel, start_idx=8001):
    output_path = os.path.join(config.model_dir, 'predicted_result.txt')
    with open(output_path, 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


def train(model, criterion, loader, config):
    train_loader, dev_loader, _ = loader
    optimizer = optim.Adadelta(
        model.parameters(), lr=config.lr, weight_decay=config.L2_decay)
    # scheduler = StepLR(optimizer,step_size=10,gamma=0.1)

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval(config)
    max_f1 = -float('inf')

    
    # 创建存储训练记录的 CSV 文件路径
    csv_path = os.path.join(config.model_dir, f'{config.model_name}_train_metrics.csv')
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    records = []

    for epoch in range(1, config.epoch+1):
        for step, (data, label) in enumerate(train_loader):
            model.train()
            data = data.to(config.device)
            label = label.to(config.device)

            optimizer.zero_grad()   # 清空梯度
            logits = model(data)
            logits = model(data)

            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=5)     # 避免梯度爆炸
            optimizer.step()

        _, train_loss, _, _, _, _, _  = eval_tool.evaluate(model, criterion, train_loader)
        f1, eval_loss, _, micro_f1, precision, recall, accuracy = eval_tool.evaluate(model, criterion, dev_loader)

        print(f'[{epoch:03d}] train_loss: {train_loss:.3f} | '
              f'dev_loss: {eval_loss:.3f} | '
              f'micro f1 on dev: {micro_f1:.4f} | '
              f'Precision on dev: {precision:.4f} | '
              f'Recall on dev: {recall:.4f} | '
              f'Accuracy on dev: {accuracy:.4f} | '
              f'Macro F1 on dev: {f1:.4f}', end=' ')

        
        
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), os.path.join(
                config.model_dir, 'model.pkl'))
            print('>>> save models!')
        else:
            print()
    
    # 将训练记录保存到 CSV 文件
    df = pd.DataFrame(records, columns=['Epoch', 'Train Loss', 'Dev Loss', 'Micro F1', 'Precision', 'Recall', 'Accuracy', 'Macro F1'])
    df.to_csv(csv_path, index=False)
    print(f'Training metrics saved to {csv_path}')


def test(model, criterion, loader, config):
    print('--------------------------------------')
    print('Start testing ...')

    _, _, test_loader = loader
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'model.pkl')))
    eval_tool = Eval(config)

    f1, test_loss, predict_label, micro_f1, precision, recall, accuracy = eval_tool.evaluate(
        model, criterion, test_loader)

    print(f'test_loss: {test_loss:.3f} | '
          f'Micro F1 on test: {micro_f1:.4f} | '
          f'Precision on test: {precision:.4f} | '
          f'Recall on test: {recall:.4f} | '
          f'Accuracy on test: {accuracy:.4f} | '
          f'Macro F1 on test: {f1:.4f}')

    # 追加测试集结果到 CSV
    csv_path = os.path.join(config.model_dir, f'{config.model_name}_test_metrics.csv')
    test_record = pd.DataFrame([['Test', None, test_loss, micro_f1, precision, recall, accuracy, f1]], columns=['Epoch', 'Train Loss', 'Dev Loss', 'Micro F1', 'Precision', 'Recall', 'Accuracy', 'Macro F1'])
    test_record.to_csv(csv_path, mode='a', header=False, index=False)
    print(f'Test metrics appended to {csv_path}')

    return predict_label


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = SemEvalDataLoader(rel2id, word2id, config)

    train_loader, dev_loader = None, None
    if config.mode == 1:  # train mode
        train_loader = loader.get_train()
        dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finish!')

    print('--------------------------------------')
    if config.model_name == 'BLSTM':
        model = BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'Att_BLSTM':
        model = Att_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'BLSTM_CNN':
        model = BLSTM_CNN(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'Att_BLSTM_CNN':
        model = Att_BLSTM_CNN(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'Multi_Att_BLSTM':
        model = Multi_Att_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
        

    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()

    if config.mode == 1:  # train mode
        train(model, criterion, loader, config)
    predict_label = test(model, criterion, loader, config)
    print_result(predict_label, id2rel)
