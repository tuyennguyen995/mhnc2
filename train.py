
import os
import matplotlib.pyplot as plt
import PIL

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense


# Build model
model = Sequential()

# add model layers
model.add(Flatten(input_shape=(92, 112, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(40, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Chuẩn hóa tập train
train_datagen = ImageDataGenerator(rescale=1. / 255)

# Chuẩn hóa tập validation
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Lấy data
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(92, 112),
    color_mode='grayscale',
    class_mode='categorical')

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(92, 112),
    color_mode='grayscale',
    class_mode='categorical')

# Tao thu muc models
path = "models/train1"

if not os.path.exists(path):
    os.makedirs(path)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=320,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=80)

# Save model
with open(path + '/report.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

model_json = model.to_json()
with open(path + "/face_model.json", "w") as json_file:
    json_file.write(model_json)
with open(path + "/face_model_weights.h5", "w") as json_file:
    model.save_weights(path +'/face_model_weights.h5')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path + '/acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path + '/loss.png')
plt.show()