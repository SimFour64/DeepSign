# TODO: Import your package, replace this by explicit imports of what you need
#from deepsign.main import predict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running! V 0.2"
    }

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
