import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# set dataset
dataset = keras.datasets.fashion_mnist
(workout_imgs, workout_identifiers), (test_img, test_identifier) = dataset.load_data()

# normalization
workout_imgs = workout_imgs / 255

def set_layers():
    layer = []
    
    layer.append(keras.layers.Flatten(input_shape=(28, 28))) #input
    layer.append(keras.layers.Dense(256, activation=tf.nn.relu)) #process/hidden
    layer.append(keras.layers.Dropout(.2)) #dropout/hidden
    layer.append(keras.layers.Dense(10, activation=tf.nn.softmax)) #output
    return layer

def fit_model(layer, epochs):
    model = keras.Sequential(layer)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", 
                  metrics=['accuracy'])

    hist = model.fit(workout_imgs, workout_identifiers, epochs=epochs, validation_split=.2)
    return hist, model

def save_model(model, file_name):
    path = 'saved_models/'
    model.save(path + file_name)
    print('Save this Model as %s' % file_name)

def load_model_file(file_model):pass

def run_model(epochs=1):
    layer = set_layers()
    model_tuple = fit_model(layer, epochs)

    return model_tuple


if __name__ == '__main__':
    run_model()