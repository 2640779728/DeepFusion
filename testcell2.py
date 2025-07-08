import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score
import model_2_fusion2
from keras.optimizers import Adam


def load_data(cell_name):
    code_file1 = f'./data/{cell_name}/4mer_datavec.csv'
    code_file2 = f'./data/{cell_name}/5mer_datavec.csv'
    code_file3 = f'./data/{cell_name}/6mer_datavec.csv'
    input_4mer = pd.read_csv(code_file1, header=None, index_col=[0])
    input_5mer = pd.read_csv(code_file2, header=None, index_col=[0])
    input_6mer = pd.read_csv(code_file3, header=None, index_col=[0])
    x1 = pd.concat([input_4mer, input_5mer, input_6mer], axis=1).values

    chip_file = f'./data/{cell_name}/chip_data.csv'
    x2 = pd.read_csv(chip_file, header=None, index_col=[0]).values

    label_file = f'./data/{cell_name}/{cell_name}.csv'
    y = pd.read_csv(label_file)
    y.loc[y.label == 'NO', 'label'] = 0
    y.loc[y.label == 'YES', 'label'] = 1
    y = y['label'].values.astype(np.int32)

    return x1, x2, y


def dataSample(x1, x2, y):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.utils import shuffle
    from collections import Counter

    X_combined = np.concatenate((x1, x2), axis=1)
    sm = SMOTE(sampling_strategy={0: int(Counter(y)[0]), 1: int(Counter(y)[1]) * 10}, random_state=42)
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_combined, y)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    n_features1 = x1.shape[1]
    x1_resampled = X_resampled[:, :n_features1]
    x2_resampled = X_resampled[:, n_features1:]
    x1_resampled, x2_resampled, y_resampled = shuffle(x1_resampled, x2_resampled, y_resampled, random_state=42)
    return x1_resampled, x2_resampled, y_resampled


def train_and_save_model(cell_name, learning_rate=0.01):
    X1_train, X2_train, y_train = load_data(cell_name)
    X1_train, X2_train, y_train = dataSample(X1_train, X2_train, y_train)

    scaler1 = StandardScaler().fit(X1_train)
    X1_train_scaled = scaler1.transform(X1_train).reshape((-1, 300, 1))
    scaler2 = StandardScaler().fit(X2_train)
    X2_train_scaled = scaler2.transform(X2_train)

    model = model_2_fusion2.get_model()
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['binary_accuracy'])
    model.fit([X1_train_scaled, X2_train_scaled], y_train, epochs=10, batch_size=64, verbose=0)

    return model, scaler1, scaler2


def average_weights(models):
    weights = [model.get_weights() for model in models]
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(np.mean(weights_list_tuple, axis=0))
    return new_weights


def evaluate_model(model, scaler1, scaler2, cell_name):
    X1_test, X2_test, y_test = load_data(cell_name)
    X1_test_scaled = scaler1.transform(X1_test).reshape((-1, 300, 1))
    X2_test_scaled = scaler2.transform(X2_test)

    y_pred_prob = model.predict([X1_test_scaled, X2_test_scaled])
    y_pred = y_pred_prob >= 0.5

    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_prob),
        'AUPR': average_precision_score(y_test, y_pred_prob),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred)
    }


if __name__ == '__main__':
    cells = ['left_lung', 'left_Ventricle', 'Pancreas', 'spleen']

    models_and_scalers = [train_and_save_model(cell) for cell in cells]
    models = [item[0] for item in models_and_scalers]
    scaler1, scaler2 = models_and_scalers[0][1], models_and_scalers[0][2]

    combined_model = model_2_fusion2.get_model()
    combined_model.set_weights(average_weights(models))

    combined_results = {}
    for cell in cells:
        combined_results[cell] = evaluate_model(combined_model, scaler1, scaler2, cell)

    print(f'Combined model results:')
    for cell, metrics in combined_results.items():
        print(f'  Test on {cell}: {metrics}')
