import joblib
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from RandomForest import read_data, read_test_data
import os


class modelMerge():
    def __init__(self, rf=None):

        self.rf = self.loadRF() if rf is None else rf
        self.probHandle = self.predict_proba

    def loadRF(self):
        rf = joblib.load('rf.pkl')
        return rf

    def predict(self, data):
        dataProb = self.rf.predict_proba(data)
        res = self.probHandle(dataProb)
        return res

    def predict_proba(self, datas):
        res = []
        for data in datas:
            # 如果data的最大值小于0.9，则label为99，否则返回data中最大值的索引+1
            if max(data) < 0.9:
                res.append(99)
            else:
                res.append(data.argmax() + 1)
        return res


if __name__ == '__main__':
    model = modelMerge()
    # data, labels = read_data()
    dataTest, labelsTest = read_test_data()
    res = model.predict(dataTest)
    print(res)
    print(labelsTest)
    print(accuracy_score(labelsTest, res))
