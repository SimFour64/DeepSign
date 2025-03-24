import os

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# répertoires de données
RAW_DATA_DIR = os.path.join(PROJECT_FOLDER, 'raw_data')
TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'sorted_signs', 'train')
TEST_DIR = os.path.join(RAW_DATA_DIR, 'sorted_signs', 'test')


#modèles
MODEL_DIR = os.path.join(PROJECT_FOLDER, 'models')

# Affichage des chemins pour vérifier
print(f"PROJECT_FOLDER: {PROJECT_FOLDER}")
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TEST_DIR: {TEST_DIR}")
