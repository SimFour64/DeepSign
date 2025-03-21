import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from params import RAW_DATA_DIR

def Folderize_images_by_category():
    """
    This fonction allows the "Folderization" of all images according
    to the category they belong to., inside a folder named with the appropriate category.
    input -->format ".npy"
    ouput -->format ".jpeg"
    """
    X = np.load("../raw_data/X.npy")  # Données d'entrainement
    Y = np.load("../raw_data/Y.npy")  # Labels (catégories)

    assert len(X) == len(Y), "Erreur: X et Y doivent avoir la même taille !"

    # Créer un dossier pour organiser les fichiers
    output_dir = "../raw_data/sorted_signs"
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir chaque catégorie unique dans Y
    for category in np.unique(Y):
        category_dir = os.path.join(output_dir, str(category))
        os.makedirs(category_dir, exist_ok=True)  # Créer le dossier si inexistant

        # Filtrer les indices appartenant à cette catégorie
        category_name = np.where(Y == category)[0]

        # conversion en jpeg
        for i, img_array in enumerate(X[category_name]):
            img_path = os.path.join(category_dir, f"image_{i}.jpeg")
            img_temp = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # inversion des couches de couleurs
            cv2.imwrite(img_path, (img_temp * 255).astype(np.uint8))  # Sauvegarde

        print(f"/////////// {len(X[category_name])} images enregistrées pour la categorie '{category}'")

    print("✅ Folderization des fichiers terminée !")
    return None

def Folderize_images_with_train_test_split(test_size=0.2):
    """
    Cette fonction permet de "folderiser" les images en fonction de leur catégorie
    et de les séparer en jeux de train et test dans les répertoires appropriés.
    input --> format ".npy"
    output --> format ".jpeg" dans des sous-répertoires 'train' et 'test'
    """
    X = np.load(os.path.join(RAW_DATA_DIR, 'X.npy'))
    Y = np.load(os.path.join(RAW_DATA_DIR, 'Y.npy'))

    assert len(X) == len(Y), "Erreur: X et Y doivent avoir la même taille !"

    # Créer les répertoires pour organiser les fichiers
    output_dir = os.path.join(RAW_DATA_DIR, 'sorted_signs')
    os.makedirs(output_dir, exist_ok=True)

    for category in np.unique(Y):
        category_train_dir = os.path.join(output_dir, 'train', str(category))
        category_test_dir = os.path.join(output_dir, 'test', str(category))
        os.makedirs(category_train_dir, exist_ok=True)
        os.makedirs(category_test_dir, exist_ok=True)

        # Filtrer les indices appartenant à cette catégorie
        category_name = np.where(Y == category)[0]

        # Séparer en jeux de train et test (80% train, 20% test)
        train_indices, test_indices = train_test_split(category_name, test_size=test_size, random_state=42)

        # Sauvegarder les images dans le répertoire train
        for i, idx in enumerate(train_indices):
            img_array = X[idx]
            img_path = os.path.join(category_train_dir, f"image_{i}.jpeg")
            img_temp = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # inversion des couches de couleurs
            cv2.imwrite(img_path, (img_temp * 255).astype(np.uint8))  # Sauvegarde

        # Sauvegarder les images dans le répertoire test
        for i, idx in enumerate(test_indices):
            img_array = X[idx]
            img_path = os.path.join(category_test_dir, f"image_{i}.jpeg")
            img_temp = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # inversion des couches de couleurs
            cv2.imwrite(img_path, (img_temp * 255).astype(np.uint8))  # Sauvegarde

        print(f"/////////// {len(train_indices)} images pour l'entraînement et {len(test_indices)} pour le test dans la catégorie '{category}'")

    print("✅ Folderization et séparation train/test terminées !")
    return None

if __name__ == '__main__':
    Folderize_images_with_train_test_split()
