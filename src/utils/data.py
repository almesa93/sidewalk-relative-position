import os
from tensorflow import keras

# Datos para la extracción de los datos
INPUT_PATH = '.\\data\\raw'

OUTPUT_PATH = '.\\data\\processed\\originals_YT'

m_sec = 4000

INPUT_PATH_HALF_IMG = OUTPUT_PATH

OUTPUT_PATH_HALF_IMG = ['.\\data\\processed\\originals_YT\\left',
                        '.\\data\\processed\\\originals_YT\\right']

class_names_label = os.listdir('.\\data\\processed\\train')

INPUT_PATHS = ['.\\data\\processed\\train',
            '.\\data\\processed\\test']

PROCESSED_INPUT_PATHS = ['.\\data\\processed\\train\\cerca_der',
                        '.\\data\\processed\\train\\cerca_izq',
                        '.\\data\\processed\\train\\lejos_der',
                        '.\\data\\processed\\train\\lejos_izq',
                        '.\\data\\processed\\train\\peligro_der',
                        '.\\data\\processed\\train\\peligro_izq']



FLIPPED_OUTPUT_PATHS = ['.\\data\\processed\\train\\cerca_izq',
                        '.\\data\\processed\\train\\cerca_der',
                        '.\\data\\processed\\train\\lejos_izq',
                        '.\\data\\processed\\train\\lejos_der',
                        '.\\data\\processed\\train\\peligro_izq',
                        '.\\data\\processed\\train\\peligro_der']

PATH_MODELS = '.\\model'

#------------------------------------------------------------------------------------------------------------

# Datos para funcionamiento de la app

model_selected_app = 'prueba_6.h5'

path_app = '.\\model\\'

m_sec_app = 1000  # milisegundos entre frame y frame







