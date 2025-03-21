from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from tensorflow.keras.models import save_model, load_model
import datetime
import os
import shutil
import random
from params import TRAIN_DIR, TEST_DIR

classes_to_select = ['hello', 'please', '2', 'c', 'NULL']

#project_dir= '/Users/veronika/code/SimFour64/DeepSign/raw_data'
#dataset_dir = 'raw_data/sorted_signs'
#test_dataset = 'test_data'


# Pourcentage d'images à garder pour le test
#test_split_ratio = 0.1  # 10% des images seront utilisées pour le test

# Création du dossier test_data s'il n'existe pas
#os.makedirs(os.path.join(project_dir,test_dataset), exist_ok=True)

# Parcourir les classes sélectionnées
#for class_name in classes_to_select:
        #source_dir = os.path.join(project_dir,dataset_dir, class_name)
        #dest_dir = os.path.join(project_dir,test_dataset, class_name)

        # Récupérer toutes les images de la classe
        #images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Sélectionner aléatoirement un pourcentage des images
        #num_test_images = int(len(images) * test_split_ratio)
        #test_images = random.sample(images, num_test_images)

        # Deplacer les images sélectionnées vers le dossier test_data
        #for image in test_images:
            #src_path = os.path.join(source_dir, image)
            #dest_path = os.path.join(dest_dir, image)
            #shutil.move(src_path, dest_path)

        #print(f"{num_test_images} images de la classe '{class_name}' copiées vers {dest_dir}")

    #print("Création du dataset de test terminée ! ✅")

#def load_test_data(test_dataset, img_size=(128, 128), batch_size=32):

    #test_dataset = image_dataset_from_directory(
        #image_size=img_size,
        #batch_size=batch_size,
        #color_mode='rgb',
        #label_mode='categorical',
        #seed=123)
    #return test_dataset


# Initialisation du modèle
def initialize_model():

    model = models.Sequential([
        # Normalisation des images (diviser par 255)
        layers.Rescaling(1./255, input_shape=(128, 128, 3)),

        # Première couche convolutionnelle avec max pooling
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        # Deuxième couche convolutionnelle avec max pooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        #layers.MaxPooling2D(2, 2),

        # Aplatir la sortie des couches convolutionnelles
        layers.Flatten(),

        # Couche dense avec 128 unités
        layers.Dense(12, activation='relu'),

        # Couche de sortie avec une unité par classe
        layers.Dense(5, activation='softmax')  # 5 classes de sortie
    ])

    return model



# Compilation du modèle
def compile_model(model):

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# Sauvegarde du modèle
def save_model(model, filename='model.keras'):
    model.save(filename)


# Entrainement du modèle
def train_model(model, epochs=5,
                img_size=(128,128), batch_size=32):

    train_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='categorical',  # Classification multi-classes
        validation_split=0.2,
        subset="training",
        seed=123,
        class_names=classes_to_select
    )

    validation_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        class_names=classes_to_select
    )


    es = EarlyStopping(monitor='val_loss',
                       patience=5,
                       restore_best_weights=True,
                       verbose=1)
    checkpoint = ModelCheckpoint(os.path.join('models','base_model.keras'),
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=[es, checkpoint]
    )
    model_name = str(datetime.datetime.now())
    save_model(model, filename=os.path.join('models',f'{model_name}.keras'))
    return history, model



# Évaluation du modèle
def evaluate_model(model, img_size=(128, 128), batch_size=32):
    test_dataset = image_dataset_from_directory(
        TEST_DIR,
        image_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='categorical',
        seed=123,
        class_names=classes_to_select
    )
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return loss, accuracy



# Afficher les courbes de performance (précision et perte)
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Courbe de précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Courbe de perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Prédiction pour une image
def predict_image(model, image):

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class


if __name__== "__main__":

    model = initialize_model()
    compile_model(model)
    history, model = train_model(model, epochs=20)
    #evaluate_model(model, test_dataset)

    plot_history(history)
    loss, accuracy = evaluate_model(model)

    # image_path = "path_to_image.jpg"
    # predicted_class = predict_image(model, image_path)
    # print(f"Predicted class: {classes_to_select[predicted_class[0]]}")
