from sklearn.preprocessing import MinMaxScaler
import numpy as np

def train_test_split(train):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train.values.reshape(-1,1))

    prediction_days = 10

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])


    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train , scaler