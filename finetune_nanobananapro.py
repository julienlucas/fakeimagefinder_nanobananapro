import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

import utils.helper_utils as helper_utils


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Définit les valeurs moyennes standard pour le dataset ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# Charge le chemin du dataset
dataset_path = "./AIvsReal_nanobanana_pro"
# Analyse les splits du dataset au chemin donné et affiche un compte des images pour chaque classe.
# helper_utils.dataset_images_per_class(dataset_path)


# Sélectionne aléatoirement et affiche une grille d'images d'échantillons depuis le dossier 'train'.
# helper_utils.display_train_images(dataset_path)

def define_transformations(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Définit des séries séparées de transformations d'images pour les datasets d'entraînement et de validation.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


def create_dataset_splits(data_path):
    """
    Crée des datasets d'entraînement et de validation à partir d'une structure de répertoires en utilisant ImageFolder.
    """
    train_path = data_path + "/train"
    val_path = data_path + "/test"

    train_dataset = ImageFolder(root=train_path)
    val_dataset = ImageFolder(root=val_path)

    return train_dataset, val_dataset


def create_data_loaders(trainset, valset, batch_size):
    """
    Crée des instances DataLoader pour les datasets d'entraînement et de validation avec leurs transformations respectives.
    """
    train_transform, val_transform = define_transformations()

    trainset.transform = train_transform
    valset.transform = val_transform

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, trainset, valset


temp_train, temp_val = create_dataset_splits(dataset_path)
dataloaders = create_data_loaders(temp_train, temp_val, batch_size=16)
# helper_utils.display_images_from_dataloader(dataloaders[0])


def load_mobilenetv3_model(weights_path, num_classes=None):
    """
    Charge un modèle MobileNetV3-Large pré-entraîné depuis torchvision.
    """
    model = tv_models.mobilenet_v3_large(weights=None)

    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model



# 1. Charge le modèle pré-entraîné
trained_model_path = "./models/best_model_midjourney_dalle_sd.pth"
trained_model = load_mobilenetv3_model(trained_model_path, num_classes=2)

# 2. Gèle toutes les features, seul le classifier sera entraîné
for param in trained_model.features.parameters():
    param.requires_grad = False

trained_model = trained_model.to(DEVICE)

print("Modèle chargé et gelé, sauf le classifier qui sera entraîné")

# 3. Prépare les deux datasets
midjourney_path = "./AIvsReal_midjourney_dalle_sd"
nanobanana_path = "./AIvsReal_nanobanana_pro"

midjourney_train, midjourney_val = create_dataset_splits(midjourney_path)
nanobanana_train, nanobanana_val = create_dataset_splits(nanobanana_path)

# Applique les transformations avant de combiner
train_transform, val_transform = define_transformations()
midjourney_train.transform = train_transform
midjourney_val.transform = val_transform
nanobanana_train.transform = train_transform
nanobanana_val.transform = val_transform

# Combine les datasets
combined_train = ConcatDataset([midjourney_train, nanobanana_train])
combined_val = ConcatDataset([midjourney_val, nanobanana_val])

# Crée les DataLoaders
new_train_loader = DataLoader(combined_train, batch_size=32, shuffle=True)
new_val_loader = DataLoader(combined_val, batch_size=32, shuffle=False)

# 4. Définit la fonction de perte
loss_fcn = nn.CrossEntropyLoss()

# 5. Configure l'optimiseur (uniquement le classifier)
optimizer = optim.Adam(trained_model.classifier.parameters(), lr=0.0005)

# 6. Entraîne le modèle
continued_model, metrics = helper_utils.training_loop_with_best_model(
    trained_model,
    new_train_loader,
    new_val_loader,
    loss_fcn,
    optimizer,
    DEVICE,
    num_epochs=5,
    model_name="best_model_nanobanana_pro.pth"
)

# 7. Trace les courbes de perte et précision
helper_utils.plot_training_metrics(metrics)

# 8. Sauvegarde les poids du modèle continué
torch.save(continued_model.state_dict(), './models/best_model_nanobanana_pro.pth')
print("Modèle continué sauvegardé : models/best_model_nanobanana_pro.pth")