## Sidewalk-relative-position-detector

<img src="https://media.istockphoto.com/photos/suburban-sidewalk-picture-id172338416?b=1&k=20&m=172338416&s=170667a&w=0&h=D39OO3Q6E6m5k_jwl3CrIDCfQu0VZvEbfdEKkP21Zy4=" alt="drawing" width="600"/>

This repository contains a prediction model that provides your relative position on the sidewalk.

Este proyecto consiste en un modelo de predicción para el reconocimiento de la posición relativa de una persona en la acera a través de los frames obtenidos de una cámara en directo.

Para ello, se entrena un modelo de redes convolucionales con miles de fotos con distintas imágenes desde distintas posiciones en la acera. Para más información, véase el fichero ‘train.py’.

El dataset se ha realizado de la siguiente forma:
1.	Extracción de los frames de vídeos grabados.
2.	Recorte de las imágenes para eliminar la información no relevante.
3.	Clasificación de las imágenes manualmente en función de la posición de la línea de la acera con respecto al margen opuesto de cada una de ellas

Una vez realizado el dataset, se han realizado las siguientes pruebas bajo las mismas condiciones en la red convolucional:
1.	En la primera prueba, se han introducido las imágenes sin realizar ningún tipo de postprocesado.
2.	La segunda prueba se llevó a cabo con una conversión de la imagen de RGB a escala de grises.
3.	En la tercera prueba, se transformaron los colores a sus negativos con la finalidad de destacar otros aspectos de las imágenes
4.	Las imágenes de la cuarta prueba se sometieron a una división en los 3 canales de color RGB que componen todas y se evaluaron de forma independiente
5.	En la quinta prueba, se hicieron dos postprocesados, primero se pasó a escala de grises y luego se obtuvieron los negativos a partir de estas imágenes.
6.	La sexta y última prueba fue simplemente mejorar el brillo y el contraste de la imagen original. Los resultados obtenidos fueron prácticamente los mismos pero con un             error bastante menor.
7.          Tras varias pruebas aumentando la complejidad de los modelos, la predicción empeoraba, por lo que la red neuronal final fue la siguiente:

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

El archivo ‘app.py’ contiene un script que, a través de una cámara, recoge frames de vídeo en directo y los manda al modelo elegido para predecir la posición dentro de la acera en la que se encuentra.
