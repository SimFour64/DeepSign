import os
import numpy as np
import cv2


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
