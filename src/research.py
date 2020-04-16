#%%
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# set dataset
dataset = keras.datasets.fashion_mnist
(workout_imgs, workout_identifiers), (test_img, test_identifier) = dataset.load_data()

if __name__ == '__main__':
    # normalization
    workout_imgs = workout_imgs / 255

    # layers
    inputLayer = keras.layers.Flatten(input_shape=(28, 28))
    processLayer = keras.layers.Dense(256, activation=tf.nn.relu)
    dropoutLayer = keras.layers.Dropout(.2)
    outputLayer = keras.layers.Dense(10, activation=tf.nn.softmax)

    # start model
    model = keras.Sequential([inputLayer, processLayer, dropoutLayer, outputLayer])
    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # fit model
    hist = model.fit(workout_imgs, workout_identifiers, epochs=5, validation_split=.2)
    print(hist)
    # predict_model = model.predict(test_img)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Acurácia por épocas')
    plt.xlabel('épocas')
    plt.ylabel('acurácia')
    plt.legend(['treino', 'validação'])
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Acurácia por épocas')
    plt.xlabel('épocas')
    plt.ylabel('acurácia')
    plt.legend(['treino', 'validação'])
    plt.show()
