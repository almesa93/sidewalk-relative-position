'''Este archivo se encarga de realizar el modelo de predicción. A lo largo del mismo, se explica los pasos que se van llevando 
a cabo para el tratamiento de los datos y la creación de dicho modelo.'''

# Librerías
from utils.functions import *
from utils.data import *

''' 
1. Obtención de los datos necesarios. En el fichero data.py tenemos los datos necesarios para usar las funciones.

1.2 Extraemos los frames de cada uno de los vídeos y reducimos su tamaño para que sean más ligeras para el modelo.

1.3 Dividimos en dos los frames obtenidos con la función anterior y nos quedamos con la parte inferior de cada mitad.
De este modo, reducimos mucho el tamaño de las imágenes y son más ligeras aún para el modelo.

1.4 Para duplicar el número de imágenes para la elaboración del modelo de predicción, hacemos un Flip de las mismas,
Así las tendremos tanto de un lado como del otro. '''

extract_and_preprocess(INPUT_PATH, OUTPUT_PATH, m_sec, INPUT_PATH_HALF_IMG, OUTPUT_PATH_HALF_IMG)

# Una vez cargadas las imágenes, se deberán clasificar manualmente entre izquierda y derecha y luego lejos, peligro y cerca.
# Una vez terminemos, contestamos 'y' y seguimos con el script.

stop_for_user()

# Para duplicar el número de datos, hacemos lo siguiente:

flip_func(PROCESSED_INPUT_PATHS, FLIPPED_OUTPUT_PATHS)

#------------------------------------------------------------------------------------------------------------
''' 2. Cargamos las imágenes de las diferentes carpetas, poniendo como clasificación 3 valores de izquierda y 3 de derecha 
(peligro_izq/peligro_der, cerca_izq/cerca_der y lejos_izq/lejos_der)'''

(train_images, train_labels), (test_images, test_labels) = load_data(INPUT_PATHS)

#------------------------------------------------------------------------------------------------------------
''' 3. Evaluamos y tratamos los datos para trabajar con ellos con convoluciones. Una vez han sido descargados y tratados como se
ha explicado en el archivo 'data.py', procedemos a su evaluación y ver si tenemos que volver a tratarlos. Para hacer las diversas
pruebas con el muestrario de imágenes del que disponemos, vamos a hacer diversos cambios para ver la forma en que mejor las
reconoce:

1ª prueba: Fotos en crudo tal como se van a importar
2ª prueba: Fotos en blanco y negro
3ª prueba: Fotos con los colores en negativo
4ª prueba: Fotos monocromáticas con cada uno de los canales de color
5ª prueba: Fotos pasadas a escala de grises y luego a negativo
6ª prueba: Fotos con mejora de contraste y brillo
'''

# Antes de esto, evaluaremos el dataset del que disponemos:

# 3.1. Dimensiones y formato del dataset:
dataset_properties(train_images)

# 3.2. Análisis gráfico:
graphs('bar', train_labels, test_labels, class_names_label)
graphs('pie', train_labels, test_labels, class_names_label)

# 3.3. Mostramos una imagen al azar para ver que todo está correcto:
print('Show a random image to check that it was correctly loaded.')
display_random_image(class_names_label, train_images, train_labels)

#------------------------------------------------------------------------------------------------------------
''' 4. Declaramos los valores de las variables para el modelo.'''

width=train_images.shape[2]
height=train_images.shape[1]
channels=train_images.shape[3]
image_size=(height, width, channels)

first_layer_conv = 64
second_layer_conv = 128
activation = 'relu'

first_layer_NN = 128
second_layer_NN = 64
final_activation = 'softmax'

batch_size = 32
epochs = 20
validation_split = 0.2

#------------------------------------------------------------------------------------------------------------
''' 5. Declaramos nuestro modelo y lo entrenamos '''

# De todas las pruebas, la número 6 fue la más satisfactoria, ya que, aunque de accuracy era muy parecida al resto, 
# el valor del error era mucho menor.

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

plot_accuracy_loss(history_6)

show_results(history_6, test_images, test_labels)

predictions = prediction(test_images, model_6)

conf_matrix(test_labels, test_images, class_names_label, model_6)

save_model(model_6, PATH_MODELS, 'prueba_6')
#------------------------------------------------------------------------------------------------------------




