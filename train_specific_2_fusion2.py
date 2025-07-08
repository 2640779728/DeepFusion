import sys
import warnings
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import keras
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import model_2_fusion2
from LogUtils import LogUtils

warnings.filterwarnings("ignore")

# create a log object
logger = LogUtils('logger').get_log()


def plot_roc_curve(y_true, y_pred, model_name, output_path):
    # 计算ROC曲线的坐标点
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - ' + model_name)
    plt.legend(loc="lower right")

    # 保存图形到文件
    plt.savefig(output_path, format='png', dpi=300)  # 可以根据需要修改格式和分辨率

    # # 显示图形
    # plt.show()


def load_data(cell_name):
    # 加载第一组特征
    code_file1 = './data/'+cell_name+'/4mer_datavec.csv'
    code_file2 = './data/'+cell_name+'/5mer_datavec.csv'
    code_file3 = './data/'+cell_name+'/6mer_datavec.csv'
    input_4mer = pd.read_csv(code_file1, header=None, index_col=[0])
    input_5mer = pd.read_csv(code_file2, header=None, index_col=[0])
    input_6mer = pd.read_csv(code_file3, header=None, index_col=[0])
    x1 = pd.concat([input_4mer, input_5mer, input_6mer], axis=1)

    # 加载第二组特征
    chip_file = './data/' + cell_name + '/chip_data.csv'
    x2 = pd.read_csv(chip_file, header=None, index_col=[0])

    # 加载标签
    label_file = './data/'+cell_name+'/'+cell_name+'.csv'
    y = pd.read_csv(label_file)
    y.loc[y.label=='NO','label'] = 0
    y.loc[y.label=='YES','label'] = 1

    return x1.values, x2.values, y['label'].values



# balanced the dataset
def dataSample(x1, x2, y):
    logger.info("doing the data sampling...")
    logger.info('Original dataset shape:%s' % Counter(y))

    # 拼接 x1 和 x2
    X_combined = np.concatenate((x1, x2), axis=1)

    # 使用SMOTE进行过采样，然后使用RandomUnderSampler进行欠采样
    sm = SMOTE(sampling_strategy={0: int(Counter(y)[0]), 1: int(Counter(y)[1]) * 10}, random_state=42)
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)

    X_resampled, y_resampled = sm.fit_resample(X_combined, y)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

    # 分离 x1 和 x2
    n_features1 = x1.shape[1]
    x1_resampled = X_resampled[:, :n_features1]
    x2_resampled = X_resampled[:, n_features1:]

    logger.info('Sampled dataset shape:%s' % Counter(y_resampled))

    # 由于重采样可能改变数据顺序，可选地对数据进行洗牌
    x1_resampled, x2_resampled, y_resampled = shuffle(x1_resampled, x2_resampled, y_resampled, random_state=42)

    return x1_resampled, x2_resampled, y_resampled


if __name__ == '__main__':
    mm_cells=['mESC_constituent','mESC','myotube','macrophage','Th-cell','proB-cell']
    hg_cells=['H2171','U87','MM1.S','spleen','left_lung','left_Ventricle','Pancreas']
    cell_name=str(sys.argv[1])
    learning_rate=float(sys.argv[2])
    epoch=int(sys.argv[3])
    logger.info('cell_name：' + cell_name)
    if cell_name in mm_cells or cell_name in hg_cells:
        t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        logger.info('Start time：' + t1)

        X1, X2, Y = load_data(cell_name)

        logger.info('learning_rate= % s, epoch = %s' % (str(learning_rate), str(epoch)))
        test_acc_list = []
        test_auc_list = []
        test_aupr_list = []
        test_recall_list = []
        test_precision_list = []

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)
        K=1
        for train_index, test_index in skf.split(np.zeros(Y.shape), Y):
            # 分别获取两组特征的训练和测试数据
            x1_train, x1_test = X1[train_index], X1[test_index]
            x2_train, x2_test = X2[train_index], X2[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            x1_train, x2_train, y_train = dataSample(x1_train, x2_train, y_train)

            # 分别对两组特征进行标准化
            scaler1 = StandardScaler().fit(x1_train)
            x1_train_scaled = scaler1.transform(x1_train)
            x1_test_scaled = scaler1.transform(x1_test)

            scaler2 = StandardScaler().fit(x2_train)
            x2_train_scaled = scaler2.transform(x2_train)
            x2_test_scaled = scaler2.transform(x2_test)

            # 调整形状以适配模型输入
            x1_train_scaled = x1_train_scaled.reshape((-1, 300, 1))
            x1_test_scaled = x1_test_scaled.reshape((-1, 300, 1))

            cnn = model_2_fusion2.get_model()
            cnn.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate),
                        metrics=['binary_accuracy'])
            train_history = cnn.fit([x1_train_scaled, x2_train_scaled], y_train, validation_data=([x1_test_scaled, x2_test_scaled], y_test),
                                    epochs=epoch, batch_size=64, verbose=2)

            # 使用两组特征进行训练集预测
            y_train_pred = cnn.predict([x1_train_scaled, x2_train_scaled])
            train_pred_class = y_train_pred >= 0.5

            # 计算并记录训练集上的性能指标
            train_acc = accuracy_score(y_train, train_pred_class)
            train_auc = roc_auc_score(y_train, y_train_pred)
            train_aupr = average_precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, train_pred_class, pos_label=1)
            train_precision = precision_score(y_train, train_pred_class, pos_label=1)

            logger.info('\r train_acc: %s train_auc: %s train_aupr: %s train_recall: %s train_precision: %s' %
                        (str(round(train_acc, 4)),
                         str(round(train_auc, 4)),
                         str(round(train_aupr, 4)),
                         str(round(train_recall, 4)),
                         str(round(train_precision, 4))))

            # 使用两组特征进行测试集预测
            y_test_pred = cnn.predict([x1_test_scaled, x2_test_scaled])
            test_pred_class = y_test_pred >= 0.5

            # 计算并记录测试集上的性能指标
            test_acc = accuracy_score(y_test, test_pred_class)
            test_auc = roc_auc_score(y_test, y_test_pred)
            test_aupr = average_precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, test_pred_class, pos_label=1)
            test_precision = precision_score(y_test, test_pred_class, pos_label=1)

            test_acc_list.append(test_acc)
            test_auc_list.append(test_auc)
            test_aupr_list.append(test_aupr)
            test_recall_list.append(test_recall)
            test_precision_list.append(test_precision)

            logger.info('\r test_acc: %s test_auc: %s test_aupr: %s test_recall: %s test_precision: %s' %
                        (str(round(test_acc, 4)),
                         str(round(test_auc, 4)),
                         str(round(test_aupr, 4)),
                         str(round(test_recall, 4)),
                         str(round(test_precision, 4))))

            model_path='./model/specific/'+cell_name+'_specific_'+str(K)+'.h5'
            cnn.save(model_path)

            # output_figure_path = f'roc/{cell_name}/roc_curve{K}.png'  # 可以根据需要修改保存路径
            # plot_roc_curve(y_test, y_test_pred, 'ROC', output_figure_path)

            K += 1

        t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        logger.info("End time：" + t2)
        # 调用函数绘制ROC图并保存

    else:
        print("Please verify your cell name!")

avg_test_acc = np.mean(test_acc_list)
avg_test_auc = np.mean(test_auc_list)
avg_test_aupr = np.mean(test_aupr_list)
avg_test_recall = np.mean(test_recall_list)
avg_test_precision = np.mean(test_precision_list)

logger.info('\r Average test_acc: %s test_auc: %s test_aupr: %s test_recall: %s test_precision: %s' % (
    str(round(avg_test_acc, 4)),
    str(round(avg_test_auc, 4)),
    str(round(avg_test_aupr, 4)),
    str(round(avg_test_recall, 4)),
    str(round(avg_test_precision, 4))
))