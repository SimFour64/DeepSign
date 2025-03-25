import os

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# répertoires de données
RAW_DATA_DIR = os.path.join(PROJECT_FOLDER, 'raw_data')
TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'sorted_signs', 'train')
TEST_DIR = os.path.join(RAW_DATA_DIR, 'sorted_signs', 'test')


#modèles
MODEL_DIR = os.path.join(PROJECT_FOLDER, 'models')


#PARAMS PROD
PROD_MODEL = 'TO DEFINE'
CLASS_NAME_5 = ['hello', 'please', '2', 'c', 'NULL']
CLASS_NAME_FULL = ['0','1','2','3','4','5','6','7','8','9','NULL','a','b','bye','c','d','e','good','good morning','hello','little bit','no','pardon','please','project','whats up','yes']

BUCKET_NAME = "deepsign_buckets"
MODEL_DIR = "models"

# Affichage des chemins pour vérifier
print(f"PROJECT_FOLDER: {PROJECT_FOLDER}")
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TEST_DIR: {TEST_DIR}")
