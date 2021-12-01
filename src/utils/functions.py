# Librerías

import cv2 as cv
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import keras
#------------------------------------------------------------------------------------------------------------


def read_images(path):
    '''
    Esta función lee la imagen de una determinada ubicación.
    Variables:
    - path: Ubicación de la imagen
    '''
    image = cv.imread(path)
    return image
#------------------------------------------------------------------------------------------------------------


def extractImages(pathIn, pathOut, ms_extract = 1000):
    '''
    Esta función extrae los frames de un vídeo en una determinada carpeta, modifica su resolución y los guarda en otra.
    Variables:
    - pathIn: Ruta donde se encuentran los vídeos
    - pathOut: Ruta donde se van a guardar los frames
    - ms_extract: milisegundos entre frame y frame capturado
    '''

    filenames = os.listdir(pathIn)
    count = 0
    for filename in filenames:
        vidcap = cv.VideoCapture(pathIn + '\\' + filename)
        print(pathIn + '\\' + filename)
        counter = 0
        while True:
            vidcap.set(cv.CAP_PROP_POS_MSEC,(counter*ms_extract))
            success,image = vidcap.read()
            if success:
                print ('Read a new frame: ', success, count)
                imagesmall = cv.resize(image, (int(image.shape[1]*0.5), int(image.shape[0]*0.5)))                                         
                cv.imwrite( pathOut + "\\frame_{}.jpg".format(count), imagesmall)
                count = count + 1
                counter = counter + 1
            else:
                print('***********************')
                print('Frames del vídeo capturados.')
                print('***********************')
                break

    return 'Todos los vídeos se han leído correctamente'
#------------------------------------------------------------------------------------------------------------


def half_pic_region(pathIn, pathOut):

    '''
    Esta función coge una imagen de una determinada ubicación, la divide en una mitad izquierda y otra derecha;
    y guarda en la misma ubicación la parte inferior de dicha foto, de modo que nos quedamos con el 70"%" de cada una de las mitades.
    Variables:
    - pathIn: Ruta donde se encuentran las imágenes
    - pathOut: Rutas de destino. Debe contener la ubicación de una carpeta para la izquierda y otra para la derecha. Es un iterable.
    '''

    filenames = os.listdir(pathIn)
    count = 0
    for file in filenames:
        try:
            path_img = pathIn + '\\' + file
            image = read_images(path_img)
            height = image.shape[0]
            width = image.shape[1]
            polygon_left = np.array([[(int(0), height), (int(width/2), height), (int(width/2), int(height*0.30)), (int(0), int(height*0.30))]])
            polygon_right = np.array([[(int(width/2), height), (int(width), height), (int(width), int(height*0.30)), (int(width/2), int(height*0.30))]])
            mask_1 = np.zeros_like(image, dtype='uint8')
            mask_2 = np.zeros_like(image, dtype='uint8')
            cv.fillPoly(mask_1, polygon_left, (255, 255, 255)) #255 color (blanco)
            cv.fillPoly(mask_2, polygon_right, (255, 255, 255))
            masked_image_left = cv.bitwise_and(image, mask_1)
            masked_image_right = cv.bitwise_and(image, mask_2)
            cropped_image_left = masked_image_left[int(height*0.30):height, 0:int(width/2)]
            cropped_image_right = masked_image_right[int(height*0.30):height, int(width/2):int(width)]
            cv.imwrite(pathOut[0] + '\\frame_left%d.jpg' % count, cropped_image_left)
            cv.imwrite(pathOut[1] + '\\frame_right%d.jpg' % count, cropped_image_right)
            count = count + 1
        
        except:
            print('***********************')
            print('Todas las imágenes divididas.')
            print('***********************')

    return 'Todas las imágenes se han dividido correctamente.'
#------------------------------------------------------------------------------------------------------------


def flip_func(pathsIn, pathsOut):

    '''
    Esta función recibe las ubicaciones de fotografías y les da la vuelta como si fuera un espejo.
    Variables:
    - pathsIn: Ubicación donde se encuentran las imágenes. Es un iterable
    - pathsOut: Ubicación donde se envían las imágenes. Es un iterable
    '''

    for i in range(len(pathsIn)):
        filenames = os.listdir(pathsIn[i])
        count = 0
        for file in filenames:
            if file not in os.listdir(pathsOut[i]):
                try: 
                    print('Flipping: ', count+1)
                    path_img = pathsIn[i] + '\\' + file
                    image = read_images(path_img)
                    flip = cv.flip(image, 1)
                    cv.imwrite(pathsOut[i] + '\\' + file, flip)
                    count = count + 1
                except:
                    break
            else:
                pass

        print('***********************')
        print('Imágenes de la carpeta {} fippeadas.'.format(pathsIn[i]))
        print('***********************')
#------------------------------------------------------------------------------------------------------------

def dataset_properties(images):

    '''
    Esta función recibe un array de imágenes y devuelve las propiedades .ndim, .shape y .size
    Variables:
     - images: iterable con las imágenes a evaluar
    '''

    print("Dimensiones:",images.ndim)
    print("Shape:",images.shape)
    print("Size:",images.size)
#------------------------------------------------------------------------------------------------------------


def resize_img(images, dim_1, dim_2):

    '''
    Esta función recibe una imagen y la devuelve con las dimensiones especificadas.
    '''

    resized_images = []

    for image in images:
        resized = cv.resize(image, (dim_1, dim_2))
        resized_images.append(resized)
    
    return np.array(resized_images)
#------------------------------------------------------------------------------------------------------------


def extract_and_preprocess(pathIn_extract, pathOut_extract, ms_extract, pathIn_half, pathOut_half):

    '''
    Esta función recibe los parámetros necesarios para extraer y preprocesar los frames de un vídeo.
    Variables:
    - pathIn_extract: Ruta donde se encuentran los vídeos
    - pathOut_extract: Ruta donde se van a guardar las imágenes
    - ms_extract: milisegundos entre frame y frame capturado
    - pathIn_half: Ruta donde se encuentran las imágenes
    - pathOut_half: Rutas de destino. Debe contener la ubicación de una carpeta para la izquierda y otra para la derecha. Es un iterable.
    - pathsIn_flip: Ubicación donde se encuentran las imágenes ya divididas. Es un iterable
    - pathsOut_flip: Ubicación donde se envían las imágenes. Es un iterable
    '''
    extractImages(pathIn_extract, pathOut_extract, ms_extract)
    half_pic_region(pathIn_half, pathOut_half)
#------------------------------------------------------------------------------------------------------------


def color2gray(images):

    '''
    Esta función recibe las imágenes en un iterable y las devuelve en escala de grises.
    Variables:
    - images: imágenes. Es un iterable.
    '''

    grey_images = []

    for image in images:
        grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        grey_images.append(grey)
    
    return np.array(grey_images)
#------------------------------------------------------------------------------------------------------------


def edit_image(images, alpha = 1, beta = 0):
    
    '''
    Esta función permite modificar el contraste y el brillo de las imágenes recibidas.
    - alpha: (1.0-3.0). Controla el contraste. Más valor, más contraste
    - beta: (0-100). Controla el brillo. Más valor, más brillo
    '''
    edited_images = []

    for image in images:

        adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
            
        edited_images.append(adjusted)
    
    return np.array(edited_images)
#------------------------------------------------------------------------------------------------------------


def negative_colors(images): 
    
    '''
    Esta función recibe las fotografías en un iterable y las devuelve con los colores en negativo.
    Variables:
    - images: imágenes. Es un iterable.
    '''

    negative_images = []

    for image in images:
        negative = 255 - image
        negative_images.append(negative)
    
    return np.array(negative_images)
#------------------------------------------------------------------------------------------------------------


def monocolor(images, color = 'blue'):
    
    '''
    Esta función recibe las fotografías en un iterable y el color (rojo, verde o azul) y las devuelve en dicho color.
    Cambia a azul ('blue') si no se especifica.
    Variables:
    - images: imágenes. Es un iterable.
    - color: canal de color que queremos sacar (red, green, blue).
    '''
    monocolor_images = []

    for image in images:

        if color == 'blue':
            b = image.copy()
            b[:,:,0] = b[:,:,1] = 0
            channel = b
        
        elif color == 'green':
            g = image.copy()
            g[:,:,0] = g[:,:,2] = 0
            channel = g
        
        elif color == 'red':
            r = image.copy()
            r[:,:,1] = r[:,:,2] = 0
            channel = r
        
        else:
            print('Error: Chose the correct color.')
            break
        
        monocolor_images.append(channel)

    return np.array(monocolor_images)
#------------------------------------------------------------------------------------------------------------


def load_data(pathIn):
    '''
    Esta función carga las imágenes de una lista de ubicaciones concretas y los asigna a las variables 'images' y 'labels', de modo que retorna
    los 'train' y los 'test' de cada una de ellas en variables distintas. El return serían (X_train, y_train), (X_test, y_test)
    Variables:
    pathIn: Lista con todas las ubicaciones. Es un iterable
    '''
    class_names = os.listdir(pathIn[0])
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

    output = []
    
    # Iterate through training and test sets
    for dataset in pathIn:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder.
            for file in os.listdir(os.path.join(dataset, folder)):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                # Open and resize the img
                image = cv.imread(img_path)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (int(image.shape[1]*0.6), int(image.shape[0]*0.6))) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output
#------------------------------------------------------------------------------------------------------------


def conv_model(layers, optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']):  
    '''
    Esta función recibe las layers, optimizer, loss y metrics necedasrios para crear una red convolucional.
    Variables:
    - layers: capas de la red convolucional y de la red neuronal
    '''


    model = keras.Sequential(layers)

    model.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics
                )
    
    return model
#------------------------------------------------------------------------------------------------------------


def model_train(model, train_images, train_labels, batch_size= 64, epochs= 15, validation_split= 0.1):
    '''
    Esta función entrena el modelo que se le manda y devuelve el history del modelo entrenado.
    Variables:
    - model: modelo de predicción. Debe ser una red neuronal.
    - train_images: imágenes para entrenar el modelo. Es un iterable.
    - train_labels: targets para entrenar el modelo. Es un iterable
    - batch_size: parámetro de entrenamiento de las redes neuronales.
    - epoch: parámetro de entrenamiento de las redes neuronales.
    - validation_split: parámetro de entrenamiento de las redes neuronales.
    '''
    train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
    train_images = train_images/255

    history = model.fit(train_images,
                    train_labels,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = validation_split
                    )
    return history
#------------------------------------------------------------------------------------------------------------


def graphs(type, train_labels, test_labels, class_names):

    '''
    Esta función recibe las imágenes de los "train" y los "label", además de las clases en las que se dividen cada una, para dibujar
    dos tipos de gráfico a elegir: de barras o de tarta.
    Variables:
    - type: tipo de gráfico. 'bar' o 'pie'
    - train_labels: Variables para test. Un iterable con las imágenes.
    - test_labels: Variables para test. Un iterable con las label.
    - class_names: las labels que existen. Es un iterable
    '''

    _, train_counts = np.unique(train_labels, return_counts=True)
    _, test_counts = np.unique(test_labels, return_counts=True)

    if type == 'bar':
        pd.DataFrame({'train': train_counts,
                'test': test_counts},
                index=class_names).plot(kind='bar',alpha=0.75, rot=45)
    
    elif type == 'pie':
        plt.pie(train_counts, explode = (0,0,0,0,0,0), labels = class_names, autopct='%1.1f%%')
        plt.axis('equal')
        plt.title('Proporción de observaciones')
        plt.show()
    
    else:
        print('Error: Insert the correct plot')
#------------------------------------------------------------------------------------------------------------


def display_random_image(class_names, images, labels):
    '''
    Saca por pantalla la imagen sobre unos ejes con su número de índice y el "label" que le corresponde.
    Variables:
    - class_names: las labels que existen. Es un iterable
    - images: Un iterable con las imágenes
    - labels: las labels de esas imágenes. Es un iterable
    '''
    images = images/255
    index = np.random.randint(images.shape[0])
    plt.imshow(images[index])
    plt.title('Image #{} : '.format(index) + class_names[labels[index]])
    plt.show()
#------------------------------------------------------------------------------------------------------------


def plot_accuracy_loss(history):
    '''
        Crea un gráfico donde se ve la "accuracy" y otro con el "loss" a lo largo de las iteraciones.
        Variables:
        - history: el histórico del modelo entrenado
    '''

    fig = plt.figure(figsize=(15,10))

    #Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label = 'acc')
    plt.plot(history.history['val_accuracy'], 'ro--', label = 'val_acc')
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    #Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'], 'bo--', label = 'loss')
    plt.plot(history.history['val_loss'], 'ro--', label = 'val_loss')
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.show()
#------------------------------------------------------------------------------------------------------------
def prediction(test_images, model):

    '''
    Esta función recibe el modelo y las imágenes que se quieren predecir.
    Variables:
    - test_images: imágenes que se quieren predecir. Es un iterable
    - model: modelo de predicción
    '''

    prediction = model.predict(test_images/255)
    print(prediction)

    return model.predict(prediction)

#------------------------------------------------------------------------------------------------------------


def conf_matrix(test_labels, test_images, class_names, model):
    
    '''
        Crea una matriz de confusión sobre un mapa de calor.
        Variables:
        - test_labels: las clases reales de las imágenes evaluadas. Es un iterable
        - pred_labels: las clases originadas por el modelo. Es un iterable
        - class_names: las clases. Es un iterable
    '''

    test_images_edited = edit_image(test_images, alpha = 1.5, beta = 20)
    predictions = model.predict(test_images_edited/255)
    pred_labels = np.argmax(predictions, axis=1)
    CM = confusion_matrix(test_labels, pred_labels)
    plt.figure(figsize=(10,8))
    ax = plt.axes()
    sns.heatmap(CM, annot=True, annot_kws={'size':10}, xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion matrix')
    plt.show()
#------------------------------------------------------------------------------------------------------------


def show_results(model, test_images, test_labels):

    '''
    Esta función recibe un modelo entrenado, las test_images y las test_labels que no se han usado para el entrenamiento del
    modelo y muestra los valores de loss y accuracy.
    Variables:
    - model: modelo entrenado
    - test_images: imágenes reservadas para el test que no se han usado para el entrenamiento del modelo. Es un iterable
    - test_labels: labels de las imágenes del test. Es un iterable
    '''

    results = model.evaluate(test_images/255, test_labels)
    print('Test loss:', results[0])
    print('Test accuracy:', results[0])
#------------------------------------------------------------------------------------------------------------


def save_model(model, path, model_name):

    '''
    Guarda el modelo enviado en la ubicación indicada a la función.
    Variables:
    - model: el modelo
    - path: ubicación para guardarlo
    - model_name: nombre del modelo
    '''

    model.save(path + '\\' + model_name + '.h5')
#------------------------------------------------------------------------------------------------------------


def load_model(path):

    '''
    Carga el modelo de la ubicación indicada.
    Variables:
    - path: ubicación del modelo.
    '''

    model = keras.models.load_model(path)

    return model
#------------------------------------------------------------------------------------------------------------


def processer(image):

    '''
    Esta función recibe una imagen (independientemente de su tamaño) y la procesa de tal modo que se adapte al modelo elegido.
    Este caso, reduce su tamaño, las divide en dos, recorta las partes necesarias, mejora su brillo y contraste y devuelve cada
    una de las partes.
    Variables:
    - image: imagen para procesar.
    '''

    height = image.shape[0]
    width = image.shape[1]
    polygon_left = np.array([[(int(0), height), (int(width/2), height), (int(width/2), int(height*0.30)), (int(0), int(height*0.30))]])
    polygon_right = np.array([[(int(width/2), height), (int(width), height), (int(width), int(height*0.30)), (int(width/2), int(height*0.30))]])
    mask_1 = np.zeros_like(image, dtype='uint8')
    mask_2 = np.zeros_like(image, dtype='uint8')
    cv.fillPoly(mask_1, polygon_left, (255, 255, 255)) #255 color (blanco)
    cv.fillPoly(mask_2, polygon_right, (255, 255, 255))
    masked_image_left = cv.bitwise_and(image, mask_1)
    masked_image_right = cv.bitwise_and(image, mask_2)
    cropped_image_left = masked_image_left[int(height*0.30):height, 0:int(width/2)]
    cropped_image_right = masked_image_right[int(height*0.30):height, int(width/2):int(width)]
    edited_cropped_image_left = edit_image(cropped_image_left, alpha = 1.5, beta = 20)
    edited_cropped_image_right = edit_image(cropped_image_right, alpha = 1.5, beta = 20)
    edited_cropped_image_left_resized = cv.resize(edited_cropped_image_left, (192, 151))
    edited_cropped_image_right_resized = cv.resize(edited_cropped_image_right, (192, 151))

    return np.array([edited_cropped_image_left_resized])/255, np.array([edited_cropped_image_right_resized])/255
#------------------------------------------------------------------------------------------------------------


def app(model, m_sec, class_names_label):

    '''
    Esta función ejecuta la app que predice en tiempo real las imágenes recibidas.
    Variables:
    - model: modelo entrenado
    - m_sec: milisegundos entre frame y frame capturados
    '''

    count = 0
    capture = cv.VideoCapture(0) # Cambiar a 2 para una webcam externa
    while True:
        capture.set(cv.CAP_PROP_POS_MSEC,(count*m_sec))
        success,image = capture.read()
        cv.imshow("test", image)
        left_image, right_image = processer(image)
        #print('Position left:', model.predict(left_image))
        #print('Position right:', model.predict(right_image))
        print(class_names_label[np.where(model.predict(left_image)[0] == model.predict(left_image)[0].max())[0][0]])
        print(class_names_label[np.where(model.predict(right_image)[0] == model.predict(right_image)[0].max())[0][0]])
        count = count + 1
        if cv.waitKey(20) & 0xFF==ord('q'): #si la letra q está pulsada sale del programa
            break

    capture.release()

    cv.destroyAllWindows()

def stop_for_user():
    print('Now, you shold classify manually the preprocessed pictures.')

    while True:
        answer=input('Did you finished? (y/n): ')

        if answer == 'y':
            break
        
        elif answer == 'n':
            print('You must classify the pictures before you continue')
        
        else:
            print('Say y/n please')
    
    print('Lets countinue')