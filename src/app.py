'''Este archivo es un predictor en directo. Toma un frame cada cierto tiempo de una cámara, en este caso la webcam, y lo manda al
modelo para que devuelva su predicción, de tal modo que sea una aplicación de procesamiento en directo.'''


# Librerías
from utils.functions import load_model, app
from utils.data import path_app, model_selected_app, m_sec_app, class_names_label

model_app = load_model(path_app + model_selected_app)

app(model_app, m_sec_app, class_names_label)