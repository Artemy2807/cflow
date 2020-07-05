# подключаем необходимые пакеты
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

# Листы
# Лист с данными для обучения и для тестирования
data = []
# Лист с ответами нейронной сети на обучающие данные
labels = []
# Возможные классы для классификации
classes = ['00000', '00001', '00002']

# Константы 
# Размер изображений для обучения и тестирования нейронной сети
IMG_SIZE = 48
# Инициализируем скорость обучения
INIT_LR = 0.01
# Общее число эпох для обучения
EPOCHS = 20
# Размер батча для обучения нейронной сети
BATCH_SIZE = 16
# Количество выходных нейроннов
OUTPUT_NUM = len(classes)

# Функции 
def lr_shedule(epoch):
    return INIT_LR*(0.1**int(epoch/10))

# Определим архитектуру нейронной сети Keras
def cnn_model():
    # Инициализируем нейронную сеть
    # Первый слой нейронной сети
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Второй слой нейронной сети
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Третий слой нейронной сети
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Полносвязный слой
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))
    return model

# Сохранение модели в формате .pb 
def save_pb(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# Создаём парсер аргументов и передаём их
ap = argparse.ArgumentParser()
# Путь до папки с фотографиями
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
help="path to output trained model")
args = vars(ap.parse_args())

# Загружаем данные 
for folder in os.listdir(args["dataset"]): 
    if folder in classes:
        for img_name in os.listdir(f'{args["dataset"]}/{folder}/'): 
            img_path = f'{args["dataset"]}/{folder}/{img_name}' 
            try:
                img = cv2.imread(img_path)
                # Изменяем размер
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # обавляем фотографию в лист
                data.append(img)
                # Получаем класс тзображения
                label = int(img_path.split('/')[-2])
                # Добавляем метку в лист
                labels.append(label) 
            except (IOError, OSError):
                print('missed', img_path) 
  
# Конвертируем листы в массивы
data = np.array(data, dtype='float32')/255.0
labels = np.eye(len(classes), dtype='uint8')[labels]

# Разбиваем данные на обучающую и тестовую выборки, используя 75%
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
labels, test_size=0.25, random_state=42)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                        featurewise_std_normalization=False,
                                                        width_shift_range=0.1,
                                                        height_shift_range=0.1,
                                                        zoom_range=0.2,
                                                        shear_range=0.1,
                                                        rotation_range=10.,)
datagen.fit(trainX)

# Конвертируем метки из целых чисел в векторы (для 2х классов при
# бинарной классификации вам следует использовать функцию Keras
# “to_categorical” вместо “LabelBinarizer” из scikit-learn, которая
# не возвращает вектор)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Создаём новую нейронную сеть и определяем архитектуру
model = cnn_model()

# Компилируем модель, используя SGD как оптимизатор 
# в качестве функции потерь (для бинарной классификации
# следует использовать binary_crossentropy)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# Обучаем нейросеть
model.fit_generator(datagen.flow(trainX, trainY, batch_size=BATCH_SIZE), epochs=EPOCHS, 
    steps_per_epoch=trainX.shape[0], validation_data=(testX, testY), callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_shedule)])

# Сохраняем модель и бинаризатор меток на диск
print("[INFO] saving network...")

# Конвертируем модель в формат .pb
frozen_graph = save_pb(tf.compat.v1.keras.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])

# Сохраняем модель на диск в формате .pb
tf.io.write_graph(frozen_graph, "./", args["model"], as_text=False)