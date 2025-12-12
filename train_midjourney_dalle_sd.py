import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import utils.helper_utils as helper_utils
import utils.unittests as unittests


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Charge le chemin du dataset
dataset_path = "./AIvsReal_sampled"
# Analyse les splits du dataset au chemin donné et affiche un compte des images pour chaque classe.
helper_utils.dataset_images_per_class(dataset_path)


# Sélectionne aléatoirement et affiche une grille d'images d'échantillons depuis le dossier 'train'.
helper_utils.display_train_images(dataset_path)


def create_dataset_splits(data_path):
    """
    Crée des datasets d'entraînement et de validation à partir d'une structure de répertoires en utilisant ImageFolder.

    Args:
        data_path (str): Le chemin racine vers le répertoire du dataset, qui devrait
                         contenir les sous-répertoires 'train' et 'validation/test'.

    Returns:
        tuple: Un tuple contenant le train_dataset et le validation_dataset
               (train_dataset, validation_dataset).
    """

    # Construit le chemin complet vers le répertoire des données d'entraînement.
    train_path = data_path + "/train"
    # Construit le chemin complet vers le répertoire des données de validation.
    val_path = data_path + "/test"

    # Crée le dataset d'entraînement en utilisant ImageFolder
    train_dataset = ImageFolder(
        # Définit la racine au chemin du dataset d'entraînement
        root=train_path,
    )

    # Crée le dataset de validation en utilisant ImageFolder
    val_dataset = ImageFolder(
        # Définit la racine au chemin du dataset de validation
        root=val_path,
    )

    return train_dataset, val_dataset


# Vérifie que la fonction charge les datasets
temp_train, temp_val = create_dataset_splits(dataset_path)
print("--- Dataset d'entraînement ---")
print(temp_train)
print("\n--- Dataset de validation ---")
print(temp_val)

# Tester le code!
unittests.exercise_1_fakefinder_transferlearning(create_dataset_splits)


# Définit les valeurs moyennes standard pour le dataset ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])

# Définit les valeurs standard de déviation pour le dataset ImageNet
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def define_transformations(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Définit des séries séparées de transformations d'images pour les datasets d'entraînement et de validation.

    Args:
        mean (list or tuple): Les valeurs moyennes (pour chaque canal, ex: RGB) calculées depuis ImageNet.
        std (list or tuple): Les valeurs d'écart-type (pour chaque canal) calculées depuis ImageNet.

    Returns:
        tuple: Un tuple contenant deux objets `torchvision.transforms.Compose` :
               - Le premier pour les transformations d'entraînement.
               - Le second pour les transformations de validation.
    """

    # Crée un objet Compose pour enchaîner plusieurs transformations ensemble pour l'ensemble d'entraînement

    # Initialise 'train_transform' en utilisant transforms.Compose pour appliquer une séquence de transformations
    train_transform = transforms.Compose([
        # Redimensionne et recadre aléatoirement l'image d'entrée à 224x224 pixels
        transforms.RandomResizedCrop((224, 224)),

        # Applique un retournement horizontal aléatoire à l'image pour l'augmentation de données
        transforms.RandomHorizontalFlip(),

        # Change aléatoirement la luminosité et le contraste de l'image pour l'augmentation de données
        # Définit `brightness=0.2` et `contrast=0.2`
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        # Convertit l'image PIL en un tenseur PyTorch
        transforms.ToTensor(),

        # Normalise le tenseur d'image avec les 'mean' et 'std' fournis pour normaliser le tenseur
        transforms.Normalize(mean, std),
    ])

    # Crée un objet Compose pour enchaîner plusieurs transformations ensemble pour l'ensemble de validation

    # Initialise 'val_transform' en utilisant transforms.Compose pour appliquer une séquence de transformations
    val_transform = transforms.Compose([
        # Redimensionne l'image d'entrée à 224x224 pixels
        transforms.Resize((224, 224)),

        # Convertit l'image PIL en un tenseur PyTorch
        transforms.ToTensor(),

        # Normalise le tenseur d'image avec les 'mean' et 'std' fournis pour normaliser le tenseur
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


# Crée les transformations composées
combined_transformations = define_transformations()
# Affiche les transformations composées pour vérifier la séquence des opérations
print("Transformations d'entraînement augmentées:\n")
print(combined_transformations[0])
print("\nTransformations de validation:\n")
print(combined_transformations[1])

# Tester le code!
unittests.exercise_2_fakefinder_transferlearning(define_transformations)


def create_data_loaders(trainset, valset, batch_size):
    """
    Crée des instances DataLoader pour les datasets d'entraînement et de validation avec leurs transformations respectives.

    Args:
        trainset (torch.utils.data.Dataset): Le dataset d'entraînement.
        valset (torch.utils.data.Dataset): Le dataset de validation.
        batch_size (int): Le nombre d'échantillons à charger dans chaque batch.

    Returns:
        tuple: Un tuple contenant :
            - train_loader (torch.utils.data.DataLoader): DataLoader pour l'ensemble d'entraînement.
            - val_loader (torch.utils.data.DataLoader): DataLoader pour l'ensemble de validation.
            - trainset (torch.utils.data.Dataset): Le dataset d'entraînement original avec les transformations maintenant appliquées.
            - valset (torch.utils.data.Dataset): Le dataset de validation original avec les transformations maintenant appliquées.
    """

    # Définit des transformations séparées pour les datasets d'entraînement et de validation
    # Utilise define_transformations() pour obtenir train_transform et val_transform
    train_transform, val_transform = define_transformations()

    # Applique les transformations d'entraînement directement au dataset d'entraînement en définissant l'attribut .transform
    trainset.transform = train_transform
    # Applique les transformations de validation directement au dataset de validation en définissant l'attribut .transform
    valset.transform = val_transform

    # Crée un DataLoader pour le dataset d'entraînement
    # Utilise le dataset d'entraînement transformé
    # Définit batch_size au batch_size d'entrée
    # Définit shuffle=True
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Crée un DataLoader pour le dataset de validation
    # Utilise le dataset de validation transformé
    # Définit batch_size au batch_size d'entrée
    # Définit shuffle=False
    val_loader  = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, trainset, valset


dataloaders = create_data_loaders(temp_train, temp_val, batch_size=16)
print("--- DataLoader d'entraînement ---")
helper_utils.display_data_loader_contents(dataloaders[0])
print("\n--- DataLoader de validation ---")
helper_utils.display_data_loader_contents(dataloaders[1])

# Tester le code!
unittests.exercise_3_fakefinder_transferlearning(create_data_loaders)


def load_mobilenetv3_model(weights_path, num_classes=None):
    """
    Charge un modèle MobileNetV3-Large pré-entraîné depuis torchvision.

    Args:
        weights_path (str): Le chemin du fichier vers les poids du modèle .pth sauvegardés.
        num_classes (int, optional): Nombre de classes. Si fourni, adapte le classifier avant de charger.

    Returns:
        torch.nn.Module: Un modèle MobileNetV3-Large pré-entraîné.
    """

    # Charge le modèle MobileNetV3-Large pré-entraîné sans poids pré-entraînés.
    model = tv_models.mobilenet_v3_large(weights=None)

    # Si num_classes fourni, adapte le classifier
    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    # Charge le dictionnaire d'état (poids) depuis le fichier local.
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    return model


# Charge le modèle MobileNetV3-Large pré-entraîné en utilisant les poids locaux.
local_weights = "./models/mobilenet_v3_large-8738ca79.pth"
test_model = load_mobilenetv3_model(local_weights)

# Affiche la dernière couche du classifieur du modèle chargé
print(test_model.classifier[-1])

# Tester le code!
unittests.exercise_4_1_fakefinder_transferlearning(load_mobilenetv3_model)


def update_model_last_layer(model, num_classes):
    """
    Gèle les couches de features d'un modèle pré-entraîné et remplace sa couche
    de classification finale par une nouvelle adaptée au nombre de classes spécifié.

    Args:
        model (torch.nn.Module): Le modèle pré-entraîné à modifier.
        num_classes (int): Le nombre de classes de sortie pour la nouvelle couche de classification.

    Returns:
        torch.nn.Module: Le modèle modifié avec des couches de features gelées et une nouvelle
                         couche de classification.
    """

    # Gèle les paramètres des couches de features du modèle
    # Itère à travers chaque paramètre dans model.features.parameters()
    for feature_parameter in model.features.parameters():
        # Définit l'attribut requires_grad de chaque feature_parameter à False
        feature_parameter.requires_grad = False

    # Accède à la couche de classification finale du modèle
    last_classifier_layer = model.classifier[-1]

    # Accède à l'attribut in_features de last_classifier_layer
    num_features = last_classifier_layer.in_features

    # Utilise nn.Linear pour créer une nouvelle couche Linear pour la classification avec le nombre original de
    # features d'entrée et le nombre de classes de sortie spécifié
    new_classifier = nn.Linear(in_features=num_features, out_features=num_classes)

    # Remplace la couche de classification originale par la couche nouvellement créée
    model.classifier[-1] = new_classifier

    return model


# Modifie la dernière couche du modèle MobileNetV3-Large
test_model = update_model_last_layer(test_model, num_classes=5)

# Affiche la dernière couche du classifieur du modèle modifié
print(test_model.classifier[-1])

# Teste votre code!
unittests.exercise_4_2_fakefinder_transferlearning(update_model_last_layer)



# Crée les datasets d'entraînement et de validation
train_dataset, val_dataset = create_dataset_splits(dataset_path)
# Crée les DataLoaders pour l'entraînement et la validation
train_loader, val_loader, _, __ = create_data_loaders(train_dataset, val_dataset, batch_size=32)



# Charge le modèle MobileNetV3-Large pré-entraîné et modifie sa dernière couche
local_weights = "./models/mobilenet_v3_large-8738ca79.pth"
mobilenet_model = load_mobilenetv3_model(local_weights)
mobilenet_model = update_model_last_layer(mobilenet_model, num_classes=2)

# Définit la fonction de perte pour calculer la différence entre la sortie du modèle et les vraies étiquettes
loss_fcn = nn.CrossEntropyLoss()

# Définit l'optimiseur pour mettre à jour les poids du modèle pendant l'entraînement
optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet_model.parameters()), lr=0.001)


# Définit le nombre d'époques
num_epochs = 1

# Entraîne le modèle
trained_model = helper_utils.training_loop_with_best_model(
    mobilenet_model,
    train_loader,
    val_loader,
    loss_fcn,
    optimizer,
    DEVICE,
    num_epochs
)