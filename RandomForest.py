import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

def read_data():
    data = []
    labels = []

    # 遍历文件
    folder_dir = f"/home/data/ljj/SampleClass"

    for i in range(1, 5):  # 四个子文件夹
        # 读取class_n.txt文件
        file_path = os.path.join(folder_dir, f"Class_{i}.txt")
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                line = line.strip()  # 去掉空格和换行符
                if line:
                    data_point = line.split()  # 根据空格分割参数
                    data.append([float(x) for x in data_point[1:]])
                    labels.append(i)

    return data, labels


def read_test_data():
    data = []
    labels = []

    data_path = "/home/data/gd/data/Data/Test_Data/Sample_1/Merge_PDW_Data.txt"
    label_path = "/home/data/gd/data/Data/Test_Data/Sample_1/Sorted_PDW.txt"
    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:1000]:
            line = line.strip()
            if line:
                data_point = line.split()
                data.append([float(x) for x in data_point[1:]])

    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:1000]:
            line = line.strip()
            if line:
                data_point = line.split()
                labels.append(int(data_point[2]))
    return data, labels


if __name__ == '__main__':
    # 读取数据集
    data, labels = read_data()
    dataTest, labelsTest = read_test_data()
    # 打印数据示例
    # for i in range(len(data)):
    #     print(f"Data: {data[i]}, Label: {labels[i]}")

    # 准备数据（假设X是你的输入特征，y是对应的目标值）
    X = data  # 输入特征
    y = labels  # 目标值

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    maxScore = 0
    # for i in range(1, 20, 1):
    # 创建随机森林分类器对象
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # random_state=42

    # 在训练集上训练分类器
    rf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = rf.predict(X_test)  # 预测结果

    # 计算准确率
    print(rf.predict_proba([[1, 1, 1, 1, 1, 1, 1]]))
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    joblib.dump(rf, 'rf.pkl')

    valPred = rf.predict(dataTest)

    print(valPred)
    valPredPro = rf.predict_proba(dataTest)
    print(labelsTest)
    print(valPredPro)
