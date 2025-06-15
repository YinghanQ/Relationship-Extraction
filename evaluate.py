#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.8

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def semeval_scorer(predict_label, true_label, class_num=10):
    import math
    assert true_label.shape[0] == predict_label.shape[0]
    
    # Initialize confusion matrix and auxiliary array for misclassified items
    confusion_matrix = np.zeros(shape=[class_num, class_num], dtype=np.float32)
    xDIRx = np.zeros(shape=[class_num], dtype=np.float32)
    
    # Iterate through each instance to update the confusion matrix
    for i in range(true_label.shape[0]):
        # Calculate the index for true and predicted labels
        true_idx = math.ceil(true_label[i] / 2)
        predict_idx = math.ceil(predict_label[i] / 2)
        
        # Update confusion matrix based on prediction correctness
        if true_label[i] == predict_label[i]:
            confusion_matrix[predict_idx][true_idx] += 1
        else:
            if true_idx == predict_idx:
                xDIRx[predict_idx] += 1  # Misclassified within the same class
            else:
                confusion_matrix[predict_idx][true_idx] += 1  # Misclassified across different classes

    # Calculate column and row sums for the confusion matrix
    col_sum = np.sum(confusion_matrix, axis=0).reshape(-1)
    row_sum = np.sum(confusion_matrix, axis=1).reshape(-1)
    
    # Initialize an array for F1 scores per class
    f1 = np.zeros(shape=[class_num], dtype=np.float32)

    # Calculate F1 score for each class, ignoring the 'Other' class
    for i in range(0, class_num):  
        try:
            # Precision and recall for each class
            p = float(confusion_matrix[i][i]) / float(col_sum[i] + xDIRx[i])
            r = float(confusion_matrix[i][i]) / float(row_sum[i] + xDIRx[i])
            f1[i] = (2 * p * r / (p + r))  # F1 score
        except:
            pass

    # Calculate the average macro F1 score
    actual_class = 0
    total_f1 = 0.0
    for i in range(1, class_num):
        if f1[i] > 0.0:  # Only consider classes with valid F1 scores
            actual_class += 1
            total_f1 += f1[i]
    
    try:
        macro_f1 = total_f1 / actual_class
    except:
        macro_f1 = 0.0

    return macro_f1


class Eval(object):
    def __init__(self, config):
        self.device = config.device  # Set device (CPU or GPU)

    def evaluate(self, model, criterion, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0
        
        # Disable gradient computation, entering evaluation mode
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            
            # Loop through the data loader to make predictions
            for _, (data, label) in enumerate(data_loader):
                data = data.to(self.device)  # Move data to the device
                label = label.to(self.device)  # Move labels to the device

                # Perform forward pass through the model
                logits = model(data)
                
                # Calculate the loss using the specified criterion
                loss = criterion(logits, label)
                total_loss += loss.item() * logits.shape[0]  # Accumulate loss

                # Get the predicted labels by taking the class with the highest logit
                _, pred = torch.max(logits, dim=1)  # Replace softmax with max function
                pred = pred.cpu().detach().numpy().reshape((-1, 1))  # Move to CPU and reshape
                label = label.cpu().detach().numpy().reshape((-1, 1))  # Move to CPU and reshape
                
                # Append predicted and true labels for later evaluation
                predict_label.append(pred)
                true_label.append(label)

        # Concatenate all predictions and true labels into a single array
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        
        # Calculate evaluation loss
        eval_loss = total_loss / predict_label.shape[0]

        # Calculate F1 score using the custom scorer function
        f1 = semeval_scorer(predict_label, true_label)
        
        # Calculate precision, recall, and accuracy using scikit-learn metrics
        precision = precision_score(true_label, predict_label, average='macro', zero_division=0)
        recall = recall_score(true_label, predict_label, average='macro', zero_division=0)
        accuracy = accuracy_score(true_label, predict_label)
        
        # Calculate micro F1 score
        micro_f1 = f1_score(true_label, predict_label, average='micro', zero_division=0)
        
        # Return all evaluation metrics
        return f1, eval_loss, predict_label, micro_f1, precision, recall, accuracy 
