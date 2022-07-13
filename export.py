import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.hdf5')

#weights
conv1 = model.layers[0].get_weights()[0]
conv2 = model.layers[1].get_weights()[0]
d1 = model.layers[4].get_weights()[0]
d2 = model.layers[5].get_weights()[0]

#biases
b1 = model.layers[0].get_weights()[1]
b2 = model.layers[1].get_weights()[1]
b3 = model.layers[4].get_weights()[1]
b4 = model.layers[5].get_weights()[1]

to_save = [conv1,conv2,d1,d2,b1,b2,b3,b4]

with open('demo.pkl', 'wb') as file:
  pickle.dump(to_save, file)