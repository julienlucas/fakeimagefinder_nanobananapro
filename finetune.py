import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms
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
dataloaders = create_data_loaders(temp_train, temp_val, batch_size=32)
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
trained_model_path = "./models/best_model_nanobanana_pro_1st.pth"
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

# import os
# import warnings
# import time
# import gc
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.datasets import ImageFolder
# import torchvision.models as tv_models
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, ConcatDataset, Dataset
# from datasets import load_dataset
# from huggingface_hub import HfApi
# import matplotlib.pyplot as plt
# import optuna
# import pandas as pd
# import lightning.pytorch as pl
# from lightning.pytorch.profilers import PyTorchProfiler
# from lightning.pytorch.callbacks import Callback
# from torch.profiler import schedule
# from torchmetrics import Accuracy

# import utils.helper_utils as helper_utils

# warnings.filterwarnings("ignore", category=UserWarning)
# torch.set_float32_matmul_precision('medium')


# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")

# # Définit les valeurs moyennes standard pour le dataset ImageNet
# IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
# IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# NANOBANANA_REPO = "./AIvsReal_nanobanana_pro"
# MIDJOURNEY_REPO = "./AIvsReal_midjourney_dalle_sd"

# def define_transformations(mean=IMAGENET_MEAN, std=IMAGENET_STD):
#     """
#     Définit des séries séparées de transformations d'images pour les datasets d'entraînement et de validation.
#     """
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.2),
#         transforms.RandomRotation(degrees=20),
#         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
#         transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
#         transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])

#     return train_transform, val_transform


# class HuggingFaceDataset(Dataset):
#     """Dataset wrapper pour les datasets Hugging Face"""
#     def __init__(self, hf_dataset, transform=None):
#         self.dataset = hf_dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         image = item["image"]
#         label = item["label"]

#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# # def create_dataset_splits(repo_id):
# #     """
# #     Crée des datasets d'entraînement et de validation depuis Hugging Face.
# #     """
# #     print(f"📥 Chargement du dataset {repo_id} depuis Hugging Face...")

# #     import tempfile
# #     from huggingface_hub import snapshot_download
# #     from pathlib import Path

# #     # Utilise le token si disponible (nettoie les retours à la ligne)
# #     token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
# #     if token:
# #         token = token.strip()

# #     with tempfile.TemporaryDirectory() as tmpdir:
# #         print(f"📦 Téléchargement des fichiers depuis Hugging Face...")
# #         snapshot_download(
# #             repo_id=repo_id,
# #             repo_type="dataset",
# #             local_dir=tmpdir,
# #             token=token if token else None
# #         )

# #         parquet_files = {
# #             "train": sorted(Path(tmpdir).glob("**/train-*.parquet")),
# #             "test": sorted(Path(tmpdir).glob("**/test-*.parquet"))
# #         }

# #         if not parquet_files["train"] and not parquet_files["test"]:
# #             parquet_files = {
# #                 "train": sorted(Path(tmpdir).glob("train-*.parquet")),
# #                 "test": sorted(Path(tmpdir).glob("test-*.parquet"))
# #             }

# #         if not parquet_files["train"] and not parquet_files["test"]:
# #             raise ValueError(f"Aucun fichier parquet trouvé dans {repo_id}")

# #         print(f"📁 Chargement depuis les fichiers Parquet...")
# #         dataset = load_dataset("parquet", data_files={
# #             "train": [str(f) for f in parquet_files["train"]],
# #             "test": [str(f) for f in parquet_files["test"]]
# #         })

# #     train_dataset = HuggingFaceDataset(dataset["train"])
# #     val_dataset = HuggingFaceDataset(dataset["test"])

# #     print(f"✅ Dataset chargé: {len(train_dataset)} train, {len(val_dataset)} test")

# #     return train_dataset, val_dataset


# def create_dataset_splits(data_path):
#     """
#     Crée des datasets d'entraînement et de validation à partir d'une structure de répertoires en utilisant ImageFolder.
#     """
#     train_path = data_path + "/train"
#     val_path = data_path + "/test"

#     train_dataset = ImageFolder(root=train_path)
#     val_dataset = ImageFolder(root=val_path)

#     return train_dataset, val_dataset


# def create_data_loaders(trainset, valset, batch_size):
#     """
#     Crée des instances DataLoader pour les datasets d'entraînement et de validation avec leurs transformations respectives.
#     """
#     train_transform, val_transform = define_transformations()

#     trainset.transform = train_transform
#     valset.transform = val_transform

#     train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, trainset, valset


# def load_model(weights_path=None, num_classes=None):
#     """
#     Charge un modèle MobileNetV3-Large pré-entraîné depuis les poids locaux.
#     """
#     # Par défaut, charger depuis mobilenet_v3_large-8738ca79.pth
#     default_weights = "./models/mobilenet_v3_large-8738ca79.pth"
#     weights_to_load = weights_path if weights_path else default_weights

#     model = tv_models.mobilenet_v3_large(weights=None)

#     if num_classes is not None:
#         num_features = model.classifier[-1].in_features
#         model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

#     if os.path.exists(weights_to_load):
#         state_dict = torch.load(weights_to_load, map_location=torch.device('cpu'))
#         # Filtrer uniquement les clés qui correspondent en nom et en forme
#         model_state = model.state_dict()
#         filtered_dict = {k: v for k, v in state_dict.items()
#                          if k in model_state and model_state[k].shape == v.shape}
#         model.load_state_dict(filtered_dict, strict=False)
#         print(f"✅ Poids chargés depuis: {weights_to_load}")
#     else:
#         print(f"⚠️  Fichier de poids non trouvé: {weights_to_load}, utilisation des poids aléatoires")

#     return model

# midjourney_repo = os.getenv("MIDJOURNEY_REPO", MIDJOURNEY_REPO)
# nanobanana_repo = os.getenv("NANOBANANA_REPO", NANOBANANA_REPO)

# class MobileNetV3LightningModule(pl.LightningModule):
#     """LightningModule pour MobileNetV3-Large"""
#     def __init__(self, learning_rate=0.0005, num_classes=2):
#         super().__init__()
#         self.save_hyperparameters()

#         # Charger depuis les poids locaux mobilenet_v3_large-8738ca79.pth
#         base_model = load_model(weights_path="./models/mobilenet_v3_large-8738ca79.pth", num_classes=num_classes)
#         for param in base_model.features.parameters():
#             param.requires_grad = False

#         self.model = base_model

#         self.loss_fn = nn.CrossEntropyLoss()
#         self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
#         self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

#         self.train_losses = []
#         self.train_accuracies = []
#         self.val_losses = []
#         self.val_accuracies = []

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         inputs, labels = batch
#         outputs = self(inputs)
#         loss = self.loss_fn(outputs, labels)

#         preds = torch.argmax(outputs, dim=1)
#         self.train_accuracy(preds, labels)
#         self.log("train_loss", loss, on_step=False, on_epoch=True)
#         self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         inputs, labels = batch
#         outputs = self(inputs)
#         loss = self.loss_fn(outputs, labels)

#         preds = torch.argmax(outputs, dim=1)
#         self.val_accuracy(preds, labels)
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
#         return loss

#     def on_train_epoch_end(self):
#         self.train_losses.append(self.trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item())
#         self.train_accuracies.append(self.trainer.callback_metrics.get("train_accuracy", torch.tensor(0.0)).item())

#     def on_validation_epoch_end(self):
#         self.val_losses.append(self.trainer.callback_metrics.get("val_loss", torch.tensor(0.0)).item())
#         self.val_accuracies.append(self.trainer.callback_metrics.get("val_accuracy", torch.tensor(0.0)).item())

#     def configure_optimizers(self):
#         return optim.Adam(self.model.classifier.parameters(), lr=self.hparams.learning_rate)


# class FakeImageDataModule(pl.LightningDataModule):
#     """DataModule pour les datasets fake/real"""
#     def __init__(self, midjourney_repo, nanobanana_repo, batch_size=64, num_workers=0):
#         super().__init__()
#         self.midjourney_repo = midjourney_repo
#         self.nanobanana_repo = nanobanana_repo
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.train_transform, self.val_transform = define_transformations()

#     def setup(self, stage=None):
#         midjourney_train, midjourney_val = create_dataset_splits(self.midjourney_repo)
#         nanobanana_train, nanobanana_val = create_dataset_splits(self.nanobanana_repo)

#         midjourney_train.transform = self.train_transform
#         midjourney_val.transform = self.val_transform
#         nanobanana_train.transform = self.train_transform
#         nanobanana_val.transform = self.val_transform

#         self.train_dataset = ConcatDataset([nanobanana_train])
#         self.val_dataset = ConcatDataset([nanobanana_val])

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# class ModelCheckpointCallback(Callback):
#     """Callback pour sauvegarder le meilleur modèle basé sur val_accuracy"""
#     def __init__(self, model_save_path='./models/best_model_nanobanana_pro.pth'):
#         super().__init__()
#         self.model_save_path = model_save_path
#         self.best_val_acc = 0.0
#         os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

#     def on_validation_epoch_end(self, trainer, pl_module):
#         if not trainer.sanity_checking:
#             metrics = trainer.callback_metrics
#             if "val_accuracy" in metrics:
#                 current_val_acc = metrics["val_accuracy"].item()
#                 previous_best = self.best_val_acc

#                 if current_val_acc > self.best_val_acc:
#                     self.best_val_acc = current_val_acc
#                     torch.save(pl_module.model.state_dict(), self.model_save_path)
#                     print(f"✅ Nouveau meilleur modèle sauvegardé! Val Accuracy: {current_val_acc*100:.2f}% (précédent: {previous_best*100:.2f}%)")
#                 else:
#                     print(f"⏭️  Modèle non sauvegardé. Val Accuracy: {current_val_acc*100:.2f}% (meilleur: {self.best_val_acc*100:.2f}%)")

# # Entraînement avec Lightning
# print("Modèle MobileNetV3-Large chargé et gelé, sauf le classifier qui sera entraîné")

# data_module = FakeImageDataModule(midjourney_repo, nanobanana_repo, batch_size=64, num_workers=2)
# lightning_model = MobileNetV3LightningModule(learning_rate=0.0005, num_classes=2)

# # Callback pour sauvegarder le meilleur modèle
# checkpoint_callback = ModelCheckpointCallback()

# trainer = pl.Trainer(
#     max_epochs=5,
#     accelerator="auto",
#     devices=1,
#     precision="32-true",
#     callbacks=[checkpoint_callback],
#     logger=False,
#     enable_progress_bar=True,
#     enable_model_summary=True,
#     enable_checkpointing=False
# )

# print("🚀 Démarrage de l'entraînement...")
# trainer.fit(lightning_model, data_module)

# # Récupérer les métriques
# metrics = (
#     lightning_model.train_losses,
#     lightning_model.train_accuracies,
#     lightning_model.val_losses,
#     lightning_model.val_accuracies
# )

# helper_utils.plot_training_metrics(metrics)

# # Sauvegarde de l'image des métriques
# metrics_image_path = "./training_metrics.png"
# plt.savefig(metrics_image_path, dpi=150, bbox_inches='tight')
# plt.close()
# print(f"📊 Graphique sauvegardé : {metrics_image_path}")

# print(f"🏆 Meilleur modèle sauvegardé avec Val Accuracy: {checkpoint_callback.best_val_acc*100:.2f}%")


# # Upload vers Hugging Face
# REPO_ID = os.getenv("HF_REPO_ID", "julienlucas/fake-image-detector")
# files_to_upload = [
#     ("./models/best_model_nanobanana_pro.pth", "best_model_nanobanana_pro.pth"),
#     (metrics_image_path, "training_metrics.png"),
# ]

# token = os.getenv("HF_TOKEN")
# if token:
#     token = token.strip()
# api = HfApi(token=token if token else None)

# print(f"\n📤 Upload vers {REPO_ID}...")
# for file_path, repo_path in files_to_upload:
#     if os.path.exists(file_path):
#         try:
#             api.upload_file(
#                 path_or_fileobj=file_path,
#                 path_in_repo=repo_path,
#                 repo_id=REPO_ID,
#                 repo_type="model",
#             )
#             print(f"  ✅ {repo_path}")
#         except Exception as e:
#             print(f"  ❌ Erreur {repo_path}: {e}")
#     else:
#         print(f"  ⚠️  Fichier non trouvé: {file_path}")