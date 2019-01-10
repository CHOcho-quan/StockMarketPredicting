import pandas as pd
import csv
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def predict_writeout(model, features_test):
    # predict
    predict = model.predict(features_test)

    # write csv
    with open("./predict.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["caseid", "Cover_Type"])
        for i in range(0, len(predict)):
            writer.writerow([(1 + i), predict[i]])

def get_data(filename):
    train = pd.read_csv(filename)
    data = train[['MidPrice', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']]
    label = train['MidPrice']
    # label = label.shift(-1)
    # label = label.dropna()
    data = np.array(data)
    # data = np.delete(data, -1, 0)
    label = np.array(label)
    labels = []
    datas = []
    tmp = []
    for d in range(0, len(list(label)), 30):
        if d % 10000 == 0:
            print("Iteration", d)
        if d+20 > len(list(label)):
            break
        datas.append(np.array(list(data[d])+list(data[d+1])+list(data[d+2])+list(data[d+3])+list(data[d+4])+list(data[d+5])+list(data[d+6])+list(data[d+7])+list(data[d+8])+list(data[d+9])))
        labels.append(label[d+10:d+30])

    return datas, labels

def evaluate_model(model,features,labels):
    a = np.mean(cross_val_score(model, features, labels, scoring="accuracy", cv=5))
    print(a)
    return a

def NeuralNetworkHyperTuning(features, labels):#hiddens, learning_rates, regularizations, tols, batch_sizes, features, labels):
    best_model = None
    best_acc = -1
    bestParameter = None
    result = {}

    # for hidden in hiddens:
    #     for lr in learning_rates:
    #         for reg in regularizations:
    #             for tol1 in tols:
    #                 for bs in batch_sizes:
    model = MLPRegressor(hidden_layer_sizes=(100, 50, 30), activation='relu', solver='adam',
    alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
    power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=-1, verbose=True,
    warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=1000)

    model.fit(features, labels)
    # accuracy = evaluate_model(model, features, labels)
    # result[(hidden, lr, reg, tol1, bs)] = accuracy
    # if accuracy > best_acc:
    #     best_acc = accuracy
    #     best_model = model
    #     bestParameter = (hidden, lr, reg, tol1, bs)

    # for p in result:
    #     print("Hyper parameter: ", p, "achieved accuracy: ", result[p])
    #
    # print ("Best accuracy achieved: ", best_acc, " Using Parameter: ", bestParameter)
    return model

train, label = get_data("train_data.csv")
print(train)
print(label)
test = pd.read_csv("test_data.csv")
test = test[['MidPrice', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']]
test = np.array(test)
i = 0
testing = []
for it in test:
    if i%10==9:
        testing.append(it)
    i+=1
# print(testing)
model = NeuralNetworkHyperTuning(train, label)
predict_writeout(model, testing)
