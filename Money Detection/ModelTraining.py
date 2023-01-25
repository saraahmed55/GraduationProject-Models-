
import os
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


root_train_dir = "C:/Users/EL-MAGD/Desktop/Money detection/dataset/train"
root_test_dir = "C:/Users/EL-MAGD/Desktop/Money detection/dataset/test/"
root_validation_dir = "C:/Users/EL-MAGD/Desktop/Money detection/dataset/valid"
# root_visualizaion_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/visualization"
saved_model_file_name = "earlyStopping.hdf5"
tensorboard_dir = "C:/Users/EL-MAGD/Desktop/Money detection/tensorboard"


import tensorflow as tf
from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224,224
image_scale = 1./255 # or 1./255 if you want to rescale (which should be the case)

nb_train_samples = 1672
nb_validation_samples = 560
train_steps = 100 # 1672 training samples/batch size of 32 = 52 steps. We are doing heavy data processing so put 500 here
validation_steps = 20 # 560 validation samples/batch size of 32 = 10 steps. We put 20 for validation steps
batch_size = 32
epochs = 60

def build_model():
    # constructing the model
    model = applications.mobilenet.MobileNet(weights="imagenet", include_top=False, 
                                             input_shape=(img_width, img_height, 3), pooling='avg')
    # only train the last 2 layers
    for layer in model.layers[:-10]:
        layer.trainable = False
    # Adding custom Layers
    x = model.output
    predictions = Dense(12, activation="softmax")(x)
    # creating the final model
    model_final = Model(inputs=model.input, outputs=predictions)
    
    return model_final

model_final = build_model()
# compile the model
model_final.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
model_final.summary()

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=image_scale,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    rotation_range=360)

validation_datagen = ImageDataGenerator(
    rescale=image_scale,
    fill_mode="nearest",
    zoom_range=0.3,
    rotation_range=30)

test_datagen = ImageDataGenerator(
    rescale=image_scale,
    fill_mode="nearest",
    zoom_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    root_train_dir,
    # save_to_dir = root_visualizaion_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = validation_datagen.flow_from_directory(
    root_validation_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    root_test_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")


# Class index
print("training labels are:", validation_generator.class_indices)

# Save the model according to the conditions
checkpoint = ModelCheckpoint(saved_model_file_name, monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=False,
                            mode='auto', period=1)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
#tensorboard = TensorBoard(log_dir = tensorboard_dir)
# Train the model
'''
history = model_final.fit_generator(
    train_generator,
    steps_per_epoch = train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = validation_steps,
    workers=16,
    callbacks=[checkpoint]
  )
'''
from keras.models import load_model 
model = load_model('C:\\Users\\EL-MAGD\\Desktop\\Money detection\\model\\earlyStopping - Copy.h5')

model.evaluate_generator(validation_generator)
validation_generator.class_indices

'''
from keras.models import load_model 
model = load_model('C:\\Users\\EL-MAGD\\Desktop\\Money detection\\model\\earlyStopping - Copy.h5')
print(model.evaluate(validation_generator)) #accurecy


import numpy as np

#from google.colab import files
from keras.preprocessing import image

uploaded=files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path='/content/' + fn
  img=image.load_img(path, target_size=(150, 150))
  
  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=10)
  
  print(classes[0])
'''
test_generator = model.predict(test_generator)
print(test_generator)
'''
'''