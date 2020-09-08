from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow 
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
#from utils import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

IMAGE_SIZE = [100,100]

epochs = 1
batch_size = 32

train_path = 'fruits-360-small/Training'
test_path = 'fruits-360-small/Validation'

image_files = glob(train_path + '/*/*jp*g')
test_image_files = glob(test_path + '/*/*jp*g')

folders = glob(train_path + '/*')

# plt.imshow(image.img_to_array(image.load_img(np.random.choice(image_files))))
# plt.show()

vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs=vgg.input, outputs = prediction)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

test_gen = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k

train_gen = gen.flow_from_directory(train_path, target_size = IMAGE_SIZE, shuffle = True, batch_size = batch_size)
test_gen = gen.flow_from_directory(test_path, target_size = IMAGE_SIZE, shuffle = True, batch_size = batch_size)


#r = model.fit_generator(train_gen, validation_data = test_gen, epochs = epochs, steps_per_epoch = len(image_files) // batch_size, validation_steps = len(test_image_files)// batch_size)
r = model.fit(train_gen, validation_data = test_gen, epochs = epochs, steps_per_epoch = len(image_files)//batch_size)

def make_confusion(data_path, N):
  	print("Generating confusion matrix", N)
  	predictions = []
  	targets = []
  	for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
  		p = model.predict(x)
  		p = np.argmax(p, axis=1)
  		y = np.argmax(y, axis=1)
  		predictions = np.concatenate((predictions, p))
  		targets = np.concatenate((targets, y))
  		if len(targets) >= N:
  			break

  	cm = confusion_matrix(targets, predictions)
  	return cm

confusion_matrix = make_confusion(train_path, len(image_files))
print(confusion_matrix)
validation_confusion_matrix = make_confusion(test_path, len(image_files))
print(validation_confusion_matrix)

plt.plot(r.history['loss'], label = 'train loss')
#plt.plot(r.history['val_loss'], label = 'validation loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label = 'train_acc')
plt.plot(r.history['val_accuracy'], label = 'test accuracy')
plt.legend()
plt.show()

plot_confusion_matrix(confusion_matrix, labels, title='Train confusion matrix')
plot_confusion_matrix(validation_confusion_matrix, labels, title='Validation confusion matrix')

print("and they all lived happily ever after")
