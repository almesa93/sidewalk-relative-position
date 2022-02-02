## Sidewalk-relative-position-detector

<img src="https://media.istockphoto.com/photos/suburban-sidewalk-picture-id172338416?b=1&k=20&m=172338416&s=170667a&w=0&h=D39OO3Q6E6m5k_jwl3CrIDCfQu0VZvEbfdEKkP21Zy4=" alt="drawing" width="600"/>

This project consists in a prediction model to recognize of the relative position of a person on the sidewalk through the frames obtained from a live camera.

To do this, a convolutional network model has been trained with thousands of photos from different positions on the sidewalk. For more information, see the 'train.py' file.

The dataset has been made in the following way:

1.	Extraction of recorded video frames.
2.	Crop images to remove non-relevant information.
3.	Classification of the images manually based on the position of the sidewalk line with respect to the opposite margin of each one of them.

Once the dataset has been made, the following tests have been carried out under the same conditions in the convolutional network:

1.	In the first test, the images have been entered without performing any type of post-processing.
2.	The second test was carried out with a conversion of the image from RGB to grayscale.
3.	In the third test, the colors were transformed into their negatives in order to highlight other aspects of the images.
4.	The images of the fourth test were divided into the 3 RGB color channels that make up all of them and were evaluated independently.
5.	In the fifth test, two post-processes were made, first it was changed to gray scale and then the negatives were obtained from these images.
6.	The sixth and final test was simply to improve the brightness and contrast of the original image. The results obtained were practically the same but with a much                 smaller error.
7.	After several tests increasing the complexity of the models, the prediction worsened, so the final neural network was the following:

```
first_layer_conv = 64
second_layer_conv = 128
activation = 'relu'

first_layer_NN = 128
second_layer_NN = 64
final_activation = 'softmax'

batch_size = 32
epochs = 20
validation_split = 0.2

layers = [keras.layers.Conv2D(first_layer_conv, (3,3), activation=activation, input_shape=image_size),
            keras.layers.MaxPooling2D(pool_size=(2,2)),

            keras.layers.Conv2D(second_layer_conv, (3,3), activation=activation),
            keras.layers.MaxPooling2D(pool_size=(2,2)),

            keras.layers.Flatten(),
            keras.layers.Dense(first_layer_NN, activation=activation),
            keras.layers.Dense(second_layer_NN, activation=activation),
            keras.layers.Dense(len(class_names_label), activation=final_activation)
        ]

train_images_edited = edit_image(train_images, alpha = 1.5, beta = 20)

model_6 = conv_model(layers,
                    optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy']
                    )


history_6 = model_train(model_6, 
                train_images,
                train_labels,
                batch_size = batch_size,
                epochs = epochs,
                validation_split = validation_split)
```
Al final, se obtuvo un 'accuracy' del 63% y un 'loss' de 2.3, bastante por encima de lo esperado ya que el dataset es propio y clasificado manualmente.

El archivo ‘app.py’ contiene un script que, a través de una cámara, recoge frames de vídeo en directo y los manda al modelo elegido para predecir la posición dentro de la acera en la que se encuentra.
