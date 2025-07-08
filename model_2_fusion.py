import keras
from keras import regularizers
from keras.layers import *
from keras.models import *

def get_model():
    # 第一组输入：原始的300维特征
    input1 = Input(shape=(300, 1))
    conv1 = Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu')(input1)
    conv2 = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=3)(conv2)
    conv3 = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=4)(conv3)
    pool2 = Dropout(0.5)(pool2)
    flat1 = Flatten()(pool2)

    # 第二组输入：额外的9维特征
    input2 = Input(shape=(3,))
    dense2_1 = Dense(64, activation='relu')(input2)
    dense2_2 = Dense(128, activation='relu')(dense2_1)

    # 融合两个分支的输出
    merged = Concatenate()([flat1, dense2_2])

    # 在融合后的特征上添加后续层
    dense3 = Dense(640, activation='relu', kernel_regularizer=regularizers.l2(0.01))(merged)
    dense3 = Dropout(0.25)(dense3)
    output = Dense(1, activation='sigmoid')(dense3)

    model = keras.Model(inputs=[input1, input2], outputs=output)
    print(model.summary())
    return model
