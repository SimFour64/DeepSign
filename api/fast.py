# TODO: Import your package, replace this by explicit imports of what you need
#from deepsign.main import predict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import cv2
import numpy as np
import os
import tensorflow as tf
from params import MODEL_DIR, X1, X2, Y1, Y2
from google.cloud import storage
import io
import tempfile

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
MODEL_PATH = os.path.join(MODEL_DIR, 'model5_5classes_accuracy_0.8993_Params_2781509.keras')
#MODEL_PATH = os.path.join('/Users/veronika/code/SimFour64/DeepSign/models/2025-03-19 16:56:39.228975_final.keras')

# Charger le modèle au démarrage de l'API
model = tf.keras.models.load_model(MODEL_PATH)

# Classes du modèle
class_names = ['hello', 'please', '2', 'c', 'NULL']
class_names_full = ['0','1','2','3','4','5','6','7','8','9','NULL','a','b','bye','c','d','e','good','good morning','hello','little bit','no','pardon','please','project','whats up','yes']

# Endpoint de test
@app.get("/")
def root():
    return {"message": "Hi, The API is running! V 0.5"}

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

    # Prétraitement de l'image en fonction des cas
    # Si l'image n'est pas carrée (vient de la webcam ordi): on crop sur le box OpenCV
    if cv2_img.shape[0] != cv2_img.shape[1]:
        cv2_img = cv2_img[Y1:Y2,X1:X2]

    # On resize dans tous les cas (même si déjà 128*128)
    cv2_img = cv2.resize(cv2_img, (128, 128))
    img_array = np.expand_dims(cv2_img, axis=0)  # Ajouter une dimension batch

    # Prédiction avec le modèle
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Obtenir l'indice de la classe prédite
    predicted_label = class_names[predicted_class]  # Obtenir le nom de la classe

    return {
        "filename": img.filename,
        "prediction": predicted_label,
        "probabilities": prediction.tolist()
    }


###################################
#  MODELS FROM GCP API #
###################################
storage_client = storage.Client()
BUCKET_NAME = "deepsign_buckets"
MODEL_DIR = "models"

# Liste les modèles disponibles dans GCP Storage
def list_models():
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=MODEL_DIR)
    models = [blob.name for blob in blobs]
    return models

# Endpoint pour lister les modèles disponibles
@app.get("/models")
def get_models():
    models = list_models()
    return {"models": models}


def load_model_from_gcs(model_name: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(model_name)

    model_bytes = blob.download_as_bytes()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
        tmp_file.write(model_bytes)
        tmp_file_path = tmp_file.name

    model = tf.keras.models.load_model(tmp_file_path)

    # Supprimer le fichier temporaire après le chargement
    # os.remove(tmp_file_path)

    return model


# Endpoint pour faire une prédiction avec un modèle sélectionné via GCP 5 CLASSES
@app.post("/get_image_prediction_from_gcp_model_5")
async def predict(model_name: str, img: UploadFile = File(...)):
    """
    Reçoit une image et un model en entrée et retourne la prédiction du modèle.
    """
    if not model_name:
        return {"error": "Model name is required"}

    # Charger le modèle
    model = load_model_from_gcs(model_name)

    # Lire l'image
    contents = await img.read()
    nparr = np.frombuffer(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Vérification si l'image est bien chargée
    if cv2_img is None:
        return {"error": "Impossible de charger l'image"}

    # Prétraitement de l'image
    img_cropped = cv2_img[Y1:Y2,X1:X2]
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

# Endpoint pour faire une prédiction avec un modèle sélectionné via GCP FULL CLASSES
@app.post("/get_image_prediction_from_gcp_model_full")
async def predict(model_name: str, img: UploadFile = File(...)):
    """
    Reçoit une image et un model en entrée et retourne la prédiction du modèle.
    """
    if not model_name:
        return {"error": "Model name is required"}

    # Charger le modèle
    model = load_model_from_gcs(model_name)

    # Lire l'image
    contents = await img.read()
    nparr = np.frombuffer(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Vérification si l'image est bien chargée
    if cv2_img is None:
        return {"error": "Impossible de charger l'image"}

    # Prétraitement de l'image
    img_cropped = cv2_img[Y1:Y2,X1:X2]
    img_resized = cv2.resize(img_cropped, (128, 128))  # Redimensionner à la taille du modèle
    #img_array = img_resized / 255.0  # Normalisation
    img_array = np.expand_dims(img_resized, axis=0)  # Ajouter une dimension batch


    # Prédiction avec le modèle
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Obtenir l'indice de la classe prédite
    predicted_label = class_names_full[predicted_class]  # Obtenir le nom de la classe

    return {
        "filename": img.filename,
        "prediction": predicted_label,
        "probabilities": prediction.tolist()
    }
