import pandas as pd
from torch import nn
import torch.optim as op
import torch
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split

# to count the score of predict to target 
from sklearn.metrics import r2_score
from sklearn import metrics

fitureNormalizer = Normalizer()


y_labels = ['a', 'ba', 'ca', 'da', 'fa', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'mpa', 'na', 'nca',
            'nga', 'nta', 'pa', 'ra', 'sa', 'ta', 'wa', 'ya']


class BPNNModelSquencial(nn.Module):

    def __init__(self, input_size, hiden_size, neuron_size, output_size, learning_rate, error_limit, drop_out=0.2):
        super(BPNNModelSquencial, self).__init__()

        # create the Net Structure
        self.layers = []
        self.layers.append(nn.Linear(input_size, neuron_size))
        self.layers.append(nn.Sigmoid())
        for i in range(hiden_size):
            self.layers.append(nn.Linear(neuron_size, neuron_size))
            self.layers.append(nn.Sigmoid())
        if drop_out > 0.0:
            self.layers.append(nn.Dropout(drop_out))
        self.layers.append(nn.Linear(neuron_size, output_size))
        self.layers.append(nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(*self.layers)

        # defaine the loss function
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = op.Adam(self.net.parameters(), lr=learning_rate)

    def train(self, x_train, y_train, epoch=1500):
        loss_value = 0
        x_train = torch.FloatTensor(x_train)
        y_train = torch.tensor(y_train)
        predict_value = []
        start = datetime.now()
        print("On Training Start at ", start)
        for i in range(epoch):
            self.net.train()
            self.optimizer.zero_grad()
            predict, predict_value = self.forward(x_train)
            loss = self.loss_func(predict, y_train)
            loss_value = loss.item()
            loss.backward()
            self.optimizer.step()
        stop = datetime.now()
        print("Training Stop at ", stop)
        # self.showPerformance(loss.item(), y_train, predict_value, start, stop)

    def forward(self, x_in):
        dataT = torch.FloatTensor(fitureNormalizer.fit_transform(x_in))
        predict = self.net(dataT)
        __, predict_value = torch.max(predict, 1)
        return predict, predict_value

    def test(self, x_test, y_test):
        x_test = torch.FloatTensor(x_test)
        y_test = torch.tensor(y_test)
        # predict with forward function
        start = datetime.now()
        pred, predict_value = self.forward(x_test)
        stop = datetime.now()
        loss = self.loss_func(pred, y_test)

        # los, pred_value = model.forward(torch.FloatTensor(x_test))
        # print("Test class size :")
        # for i in range(0,22):
        #   print(i, "Class Size : ",np.count_nonzero(y_test==i))

        # print(np.array(y_test) == np.array(pred_value))

        # print()
        # print("Matrix confusion")
        # print(confusion_matrix(np.array(y_test), np.array(predict_value)))
        # print()
        # print("Clasification Report")
        # print(classification_report(np.array(y_test), np.array(predict_value)))

        # count score
        score = 1 - loss.item()
        recall, precision, accuracy = self.getPerformance(
            y_test, predict_value)
        dur = stop - start
        print(" ", accuracy, ", ", precision, ", ",
              recall, ",", dur.total_seconds(), ";")
        # self.showPerformance(loss.item(), y_test, predict_value, start, stop)
        return accuracy, precision, recall, dur.total_seconds()

    def getPerformance(self, y_target, y_pred):
        y_target = np.array(y_target)
        y_pred = np.array(y_pred)
        recall = metrics.recall_score(y_target, y_pred, average='micro')
        precision = metrics.precision_score(y_target, y_pred, average='micro')
        accuracy = metrics.accuracy_score(y_target, y_pred)
        return recall, precision, accuracy

    def showPerformance(self, loss, y_target, y_pred, startTime, stopTime):
        recall, precision, accuracy = self.getPerformance(y_target, y_pred)
        duration = stopTime - startTime
        print(" RECALL    |=========> ", recall)
        print(" PRECISION |=========> ", precision)
        print(" ACCURACY  |=========> ", accuracy)
        print(" DURATION  |=========> ", duration.total_seconds(), "seconds")
        print()


mlModel = BPNNModelSquencial(196, 1, 64, 22, 0.01, 0.001)


def TrainModel():
    dataset = pd.read_csv("./DEF_DATASET.csv")
    # Format dataset to correct data
    x = dataset
    x = x.drop('Citra', axis=1)
    x = x.drop('Lable', axis=1)
    y = dataset[['Lable']]
    y_tresh_label = np.unique(y)
    print(y_tresh_label)
    y_labels = y_tresh_label
    labelEncode = LabelEncoder()
    x = fitureNormalizer.fit_transform(x)
    y = labelEncode.fit_transform(y)
    print("LabelEncode", y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=22, stratify=y)
    mlModel.train(x_train, y_train)
    print(mlModel.test(x_test, y_test))
