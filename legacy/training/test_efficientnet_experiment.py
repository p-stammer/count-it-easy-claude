import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

def load_model(shape=(150,150,3)):

    feature_extractor = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False,weights="imagenet", input_shape=shape)
    # feature_extractor = tf.keras.applications.MobileNetV3Small(include_top=False,weights="imagenet", input_shape=shape)
    feature_extractor.trainable = False
    cls = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor.output)
    cls = tf.keras.layers.Dropout(rate=0.2)(cls)
    cls = tf.keras.layers.Dense(90, activation = tf.nn.relu)(cls)
    cls = tf.keras.layers.Dense(90, activation = tf.nn.relu)(cls)
    cls = tf.keras.layers.Dense(9, activation = tf.nn.sigmoid)(cls)
    model = tf.keras.Model(inputs=feature_extractor.input,outputs=cls)

    # model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=True,weights=None,classes=9, input_shape=shape)
    # model = tf.keras.applications.MobileNetV3Small(include_top=True,weights=None,classes=9, input_shape=shape)

    return model

def get_num_masks():
    num_masks = np.array([[0.06729475, 0.17292225, 0.26379542, 0.31903485],
       [0.06998654, 0.36863271, 0.2705249 , 0.50670241],
       [0.08344549, 0.55898123, 0.2705249 , 0.68766756],
       [0.23149394, 0.10053619, 0.42664872, 0.21715818],
       [0.24091521, 0.27077748, 0.42934051, 0.4075067 ],
       [0.24091521, 0.45978552, 0.42799462, 0.59785523],
       [0.24899058, 0.65013405, 0.4320323 , 0.78820375],
       [0.4051144 , 0.01206434, 0.5948856 , 0.12466488],
       [0.40107672, 0.18096515, 0.58950202, 0.31635389],
       [0.41318977, 0.36193029, 0.59757739, 0.50402145],
       [0.4051144 , 0.5536193 , 0.5948856 , 0.68900804],
       [0.4064603 , 0.73726542, 0.60296097, 0.88203753],
       [0.56527591, 0.08713137, 0.76043069, 0.22117962],
       [0.5666218 , 0.26407507, 0.75235532, 0.40348525],
       [0.56527591, 0.44906166, 0.75504711, 0.59517426],
       [0.57065949, 0.64745308, 0.756393  , 0.77747989],
       [0.72543742, 0.17828418, 0.92462988, 0.30965147],
       [0.730821  , 0.36595174, 0.92866756, 0.49731903],
       [0.7294751 , 0.55093834, 0.92732167, 0.68766756],
       [0.04576043, 0.28418231, 0.20188425, 0.42091153],
       [0.18573351, 0.1849866 , 0.35935397, 0.32171582],
       [0.3526245 , 0.09383378, 0.52355316, 0.21983914],
       [0.04037685, 0.47184987, 0.20592194, 0.60589812],
       [0.1897712 , 0.36863271, 0.36473755, 0.51474531],
       [0.3512786 , 0.28016086, 0.52355316, 0.41823056],
       [0.52220727, 0.18096515, 0.68775236, 0.31635389],
       [0.0551817 , 0.6541555 , 0.21130552, 0.79356568],
       [0.20323015, 0.56300268, 0.36877524, 0.6997319 ],
       [0.36339166, 0.46112601, 0.52893674, 0.60053619],
       [0.51951548, 0.36595174, 0.69313594, 0.49731903],
       [0.68506057, 0.269437  , 0.84522207, 0.40080429],
       [0.20861373, 0.75603217, 0.36473755, 0.89544236],
       [0.3512786 , 0.6541555 , 0.53162853, 0.79490617],
       [0.52220727, 0.5536193 , 0.69582773, 0.69034853],
       [0.68640646, 0.46514745, 0.85195155, 0.58981233],
       [0.36339166, 0.84718499, 0.54643338, 0.97587131],
       [0.52624495, 0.75067024, 0.7039031 , 0.87533512],
       [0.69313594, 0.6461126 , 0.85733513, 0.78150134],
       [0.46837147, 0.09651475, 0.64064603, 0.21983914],
       [0.62987887, 0.18766756, 0.79946164, 0.31635389],
       [0.79004038, 0.27882038, 0.95693136, 0.41957105],
       [0.2960969 , 0.1849866 , 0.47779273, 0.31099196],
       [0.46567968, 0.27747989, 0.65006729, 0.40080429],
       [0.63795424, 0.36863271, 0.80753701, 0.49731903],
       [0.79946164, 0.46782842, 0.95020188, 0.61126005],
       [0.14266487, 0.28820375, 0.32032301, 0.41018767],
       [0.31763122, 0.38337802, 0.47913863, 0.50268097],
       [0.46567968, 0.46246649, 0.64333782, 0.5924933 ],
       [0.63122476, 0.55898123, 0.80619112, 0.68766756],
       [0.80215343, 0.65951743, 0.96096904, 0.80563003],
       [0.14266487, 0.47184987, 0.32032301, 0.59651475],
       [0.30551817, 0.56300268, 0.48452221, 0.68364611],
       [0.46837147, 0.65549598, 0.6487214 , 0.77211796],
       [0.64064603, 0.74932976, 0.80619112, 0.88739946],
       [0.15208614, 0.64477212, 0.3243607 , 0.78820375],
       [0.31090175, 0.74932976, 0.49125168, 0.87265416],
       [0.48048452, 0.84316354, 0.65006729, 0.99463807]])

    return num_masks

def apply_num_masks(image, num_masks):
    masked_images=[]
    offset_x = 0
    offset_y = 0
    zoom = 0
    num_masks = (num_masks*[image.shape[1],image.shape[0],image.shape[1],image.shape[0]]).astype(int) + [offset_x, offset_y, offset_x, offset_y] + [zoom,zoom,-zoom,-zoom]
    for i in range(len(num_masks)):
        masked_images.append(image[num_masks[i,1]:num_masks[i,3],num_masks[i,0]:num_masks[i,2],:])

    return masked_images


def make_images():
    path = "C:/Users/stammer/project_starfish/data/t5.jpg"
    out_dir = "C:/Users/stammer/project_starfish/data/data/train/"
    im = np.array(Image.open(path))
    # bm = np.array(np.array([0.03602258, 0.0334    , 0.96834379, 0.96573752]) * [im.shape[1],im.shape[0],im.shape[1],im.shape[0]],dtype=int)
    # im = im[bm[1]:bm[3],bm[0]:bm[2],:]
    num_images = apply_num_masks(im,get_num_masks())
    # label = [1, 9, 9, 1, 1, 1, 1, 5, 5, 5, 5, 1, 5, 5, 5, 5, 9, 9, 9, 7, 7, 7,
    #    7, 2, 2, 6, 2, 2, 2, 2, 2, 6, 6, 6, 6, 7, 7, 7, 4, 4, 8, 4, 4, 3,
    #    3, 8, 8, 8, 8, 8, 3, 3, 3, 8, 3, 4, 3] # t3.jpg
    # label = [5, 5, 5, 9, 9, 9, 9, 1, 1, 1, 1, 1, 9, 9, 9, 9, 5, 1, 5, 6, 6, 6,
    #    7, 7, 7, 7, 2, 2, 2, 2, 2, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8,
    #    3, 3, 3, 8, 3, 3, 4, 4, 4, 4, 4, 4, 4] # t4.jpg
    # label = [5,5,5,5,5,5,5,9,9,9,9,9,1,1,1,1,1,1,9,7,2,7,6,2,2,2,7,7,7,7,7,6,7,2,6,6,6,6,4,4,4,3,3,8,3,4,8,3,3,4,8,8,8,8,3,4,3] # t2.jpg
    # label = [1,1,1,5,5,5,9,9,9,9,9,9,1,5,5,1,5,5,9,2,2,2,2,6,7,2,7,7,7,7,7,7,2,2,6,6,6,6,3,4,3,4,4,4,4,3,3,3,3,3,8,8,8,8,8,8,4] # t1.jpg
    label = [5,5,5,9,9,9,9,1,1,1,1,1,9,9,9,9,5,5,5,7,7,7,7,7,7,7,2,2,2,2,2,6,6,6,6,6,6,6,4,4,4,8,8,8,8,3,3,3,3,3,4,4,4,4,8,8,8] # t5.jpg
    for i in range(len(num_images)):
        # plt.imshow(num_images[i])
        # plt.show()
        out_path = out_dir + str(label[i]) + "/"
        out_path = out_path + str(len(glob.glob(out_path+"*"))) +".jpg"
        Image.fromarray(num_images[i]).save(out_path)

def load_data(path):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                        # rotation_range=10, width_shift_range=0.2,
                                                        # height_shift_range=0.2, brightness_range=None, shear_range=0.2, zoom_range=0.2,
                                                        # fill_mode='nearest',
                                                        preprocessing_function=None)

    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    train = train_gen.flow_from_directory(path+"/train",target_size=(150, 150),
                                    batch_size=64,
                                    class_mode='categorical')
    val = val_gen.flow_from_directory(path+"/val",target_size=(150, 150),
                                    batch_size=9,
                                    class_mode='categorical')
    return train, val

def change_order(data):
    # changer = np.array([7,9,17,  22,24,26,12,  40,38,63,34,29,  42,46,60,32,  51,53,56,     6,21,39,  16,25,37,43,  19,11,35,48,50,  20,28,62,54,  14,30,58,     41,44,52,  23,47,49,55,  8,36,61,59,57,  10,27,33,31,  18,13,15])
    # changer = list(changer-6)
    data = np.array(data)
    changer = [ 1,  3, 11, 16, 18, 20,  6, 34, 32, 56, 28, 23, 36, 39, 53, 26, 44,
       46, 49,  0, 15, 33, 10, 19, 31, 37, 13,  5, 29, 41, 43, 14, 22, 55,
       47,  8, 24, 51, 35, 38, 45, 17, 40, 42, 48,  2, 30, 54, 52, 50,  4,
       21, 27, 25, 12,  7,  9]
    return data[changer]

def compile_model(model):
    [7,9,17,  22,24,26,12,  40,38,63,34,29,  42,46,60,32,  51,53,56,     6,21,39,  16,25,37,43,  19,11,35,48,50,  20,28,62,54,  14,30,58,     41,44,52,  23,47,49,55,  8,36,61,59,57,  10,27,33,31,  18,13,15]
    # []
    return model

def train_model(model, train_data, val_data, epochs, learning_rate = 1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate), loss=tf.keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    history = model.fit(train_data, epochs=epochs, validation_data=val_data,verbose=1)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate/10), loss=tf.keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    for layer in model.layers:
        layer.trainable=True
    history2 = model.fit(train_data, epochs=int(epochs), validation_data=val_data,verbose=1)
    return history

def main():
    path = "C:/Users/stammer/project_starfish/data/data/"
    model = load_model(shape=(150,150,3))
    train, val = load_data(path)
    history = train_model(model,train,val,epochs=200)

    # make_images()


if __name__ == "__main__":
    main()