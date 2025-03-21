# TODO: Import your package, replace this by explicit imports of what you need
#from deepsign.main import predict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import cv2
import numpy as np
import os
import tensorflow as tf
from params import MODEL_DIR



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)




# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(input_one: float,
            input_two: float):
    # TODO: Do something with your input
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs
    prediction = float(input_one) + float(input_two)
    return {
        'prediction': prediction,
        'inputs': {
            'input_one': input_one,
            'input_two': input_two
        }
    }

@app.get("/preprod")
def root_preprod():
    return {"message": "Hello from preprod fake API"}

@app.get("/predict_preprod")
def get_predict_preprod(input_one: float,input_two: float):
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs
    prediction = float(input_one) + float(input_two)
    return {
        'prediction': prediction,
        'inputs': {
            'input_one': input_one,
            'input_two': input_two
        }
    }

@app.post("/upload_image_preprod")
async def receive_image_preprod(img:UploadFile=File(...)):
    """
    Returning prediction from an image of american sign language
    Input: image loaded from website application
    Output : prediction from DeepSign computer vision model
    """
    # Receiving the image and decoding it
    contents = await img.read()  # Image is binarized via .read() in the frontend

    # Transforming to np.ndarray to be readable by opencv
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Response for preprod
    return {
        "filename": f"{img.filename}_processed",
        "image_size": cv2_img.shape[:-1]
    }

    ###################################
    #  OPENCV TREATMENT & PREDICTION  #
    ###################################
    # OpenCV treatment : extracting zone of interest
    # Loading production model
    # Predicting with model
    # return pred

# Répertoire des modèles
MODEL_PATH = os.path.join(MODEL_DIR, '2025-03-19 16:56:39.228975_final.keras')
#MODEL_PATH = os.path.join('/Users/veronika/code/SimFour64/DeepSign/models/2025-03-19 16:56:39.228975_final.keras')

# Charger le modèle au démarrage de l'API
model = tf.keras.models.load_model(MODEL_PATH)

# Classes du modèle
class_names = ['hello', 'please', '2', 'c', 'NULL']

# Endpoint de test
@app.get("/")
def root():
    return {"message": "Hi, The API is running! V 0.3"}

# Endpoint de prédiction d'une image
@app.post("/get_image_prediction")
async def get_prediction(img: UploadFile = File(...)):
    """
    Reçoit une image en entrée et retourne la prédiction du modèle.
    """

    # Lire l'image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Vérification si l'image est bien chargée
    if cv2_img is None:
        return {"error": "Impossible de charger l'image"}

    # Prétraitement de l'image
    img_cropped = cv2_img[200:500,200:500]
    img_resized = cv2.resize(img_cropped, (128, 128))  # Redimensionner à la taille du modèle
    #img_array = img_resized / 255.0  # Normalisation
    img_array = np.expand_dims(img_resized, axis=0)  # Ajouter une dimension batch


    # Prédiction avec le modèle
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Obtenir l'indice de la classe prédite
    predicted_label = class_names[predicted_class]  # Obtenir le nom de la classe

    return {
        "filename": img.filename,
        "prediction": predicted_label,
        "probabilities": prediction.tolist()
    }
