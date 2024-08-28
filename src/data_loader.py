import numpy as np
import os


class Scaler:
    def __init__(self):
        self.params = {}
    
    def fit(self, data):
        self.params['mean'] = np.mean(data)
        self.params['std_dev'] = np.std(data)
    
    def standardize(self, data):
        return (data - self.params['mean'])/self.params['std_dev']
    
    def fit_standardize(self, data):
        self.fit(data)
        return self.standardize(data)

    def unstandardize(self, data):
        return (data * self.params['std_dev']) + self.params['mean']
    
def one_hot_encode(data):
    data = [str(item) for item in data]
    unique_category = sorted(set(data))
    category_index = {category : index for index, category in enumerate(unique_category)}
    one_hot_matrix = []
    for item in data:
        one_hot_vector = [0] * len(unique_category)
        index = category_index[item]
        one_hot_vector[index] = 1
        one_hot_matrix.append(one_hot_vector)
    one_hot_matrix = np.array(one_hot_matrix)
    return one_hot_matrix

def prepare_train(data):
    x1 = np.array(data['male'])
    x1 = np.reshape(x1, (x1.shape[0], 1))

    x2 = np.array(data['age'])
    x2 = np.reshape(x2, (x2.shape[0], 1))
    scaler_x2 = Scaler()
    x2 = scaler_x2.fit_standardize(x2)

    x3 = np.array(data['currentSmoker'])
    x3 = np.reshape(x3, (x3.shape[0], 1))

    x4 = np.array(data['cigsPerDay'])
    x4 = np.reshape(x4, (x4.shape[0], 1))
    scaler_x4 = Scaler()
    x4 = scaler_x4.fit_standardize(x4)

    x5 = np.array(data['BPMeds'])
    x5 = np.reshape(x5, (x5.shape[0], 1))
    
    x6 = np.array(data['prevalentStroke'])
    x6 = np.reshape(x6, (x6.shape[0], 1))

    x7 = np.array(data['prevalentHyp'])
    x7 = np.reshape(x7, (x7.shape[0], 1))
    
    x8 = np.array(data['diabetes'])
    x8 = np.reshape(x8, (x8.shape[0], 1))

    x9 = np.array(data['totChol'])
    x9 = np.reshape(x9, (x9.shape[0], 1))
    scaler_x9 = Scaler()
    x9 = scaler_x9.fit_standardize(x9)

    x10 = np.array(data['sysBP'])
    x10 = np.reshape(x10, (x10.shape[0], 1))
    scaler_x10 = Scaler()
    x10 = scaler_x10.fit_standardize(x10)

    x11 = np.array(data['diaBP'])
    x11 = np.reshape(x11, (x11.shape[0], 1))
    scaler_x11 = Scaler()
    x11 = scaler_x11.fit_standardize(x11)

    x12 = np.array(data['BMI'])
    x12 = np.reshape(x12, (x12.shape[0], 1))
    scaler_x12 = Scaler()
    x12 = scaler_x12.fit_standardize(x12)

    x13 = np.array(data['heartRate'])
    x13 = np.reshape(x13, (x13.shape[0], 1))
    scaler_x13 = Scaler()
    x13 = scaler_x13.fit_standardize(x13)

    x14 = np.array(data['glucose'])
    x14 = np.reshape(x14, (x14.shape[0], 1))
    scaler_x14 = Scaler()
    x14 = scaler_x14.fit_standardize(x14)

    X15 = np.array(data['education'])
    X15 = np.reshape(X15, (X15.shape[0], 1))
    X15 = one_hot_encode(X15)

    X = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, X15]
    X = np.hstack(X)

    y = np.array(data['TenYearCHD'])
    y = np.reshape(y, (y.shape[0], 1))
    
    scalers = {'x2':scaler_x2, 'x4':scaler_x4, 'x9':scaler_x9, 'x10':scaler_x10, 'x11':scaler_x11, 'x12':scaler_x12, 'x13':scaler_x13, 'x14':scaler_x14}

    return X, y, scalers

def prepare_test(data, scalers):
    scaler_x2 = scalers['x2']
    scaler_x4 = scalers['x4']
    scaler_x9 = scalers['x9']
    scaler_x10 = scalers['x10']
    scaler_x11 = scalers['x11']
    scaler_x12 = scalers['x12']
    scaler_x13 = scalers['x13']
    scaler_x14 = scalers['x14']

    x1 = np.array(data['male'])
    x1 = np.reshape(x1, (x1.shape[0], 1))

    x2 = np.array(data['age'])
    x2 = np.reshape(x2, (x2.shape[0], 1))
    x2 = scaler_x2.fit_standardize(x2)

    x3 = np.array(data['currentSmoker'])
    x3 = np.reshape(x3, (x3.shape[0], 1))

    x4 = np.array(data['cigsPerDay'])
    x4 = np.reshape(x4, (x4.shape[0], 1))
    x4 = scaler_x4.fit_standardize(x4)

    x5 = np.array(data['BPMeds'])
    x5 = np.reshape(x5, (x5.shape[0], 1))
    
    x6 = np.array(data['prevalentStroke'])
    x6 = np.reshape(x6, (x6.shape[0], 1))

    x7 = np.array(data['prevalentHyp'])
    x7 = np.reshape(x7, (x7.shape[0], 1))
    
    x8 = np.array(data['diabetes'])
    x8 = np.reshape(x8, (x8.shape[0], 1))

    x9 = np.array(data['totChol'])
    x9 = np.reshape(x9, (x9.shape[0], 1))
    x9 = scaler_x9.fit_standardize(x9)

    x10 = np.array(data['sysBP'])
    x10 = np.reshape(x10, (x10.shape[0], 1))
    x10 = scaler_x10.fit_standardize(x10)

    x11 = np.array(data['diaBP'])
    x11 = np.reshape(x11, (x11.shape[0], 1))
    x11 = scaler_x11.fit_standardize(x11)

    x12 = np.array(data['BMI'])
    x12 = np.reshape(x12, (x12.shape[0], 1))
    x12 = scaler_x12.fit_standardize(x12)

    x13 = np.array(data['heartRate'])
    x13 = np.reshape(x13, (x13.shape[0], 1))
    x13 = scaler_x13.fit_standardize(x13)

    x14 = np.array(data['glucose'])
    x14 = np.reshape(x14, (x14.shape[0], 1))
    x14 = scaler_x14.fit_standardize(x14)

    X15 = np.array(data['education'])
    X15 = np.reshape(X15, (X15.shape[0], 1))
    X15 = one_hot_encode(X15)

    X = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, X15]
    X = np.hstack(X)

    y = np.array(data['TenYearCHD'])
    y = np.reshape(y, (y.shape[0], 1))

    return X, y
    