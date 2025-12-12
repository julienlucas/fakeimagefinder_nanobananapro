import inspect
import re
from types import FunctionType

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as tv_models
from torchvision.models import MobileNetV3

from dlai_grader.grading import test_case, print_feedback
from . import unittests_utils
from .unittests_utils import MockImageFolder


def exercise_1_fakefinder_transferlearning(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_dataset_splits a un type incorrect"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        expected_type = ImageFolder
        dataset_path = "./AIvsReal_sampled"
        train_path = dataset_path + "/train"
        val_path = dataset_path + "/test"

        learner_train, learner_val = learner_func(dataset_path)

        # Vérification du type de retour (train)
        t = test_case()
        if not isinstance(learner_train, expected_type):
            t.failed = True
            t.msg = "Type de train_dataset incorrect retourné par create_dataset_splits"
            t.want = f"{expected_type}"
            t.got = f"{type(learner_train)}"
            return [t]

        # Vérification du type de retour (val)
        t = test_case()
        if not isinstance(learner_val, expected_type):
            t.failed = True
            t.msg = "Type de val_dataset incorrect retourné par create_dataset_splits"
            t.want = f"{expected_type}"
            t.got = f"{type(learner_val)}"
            return [t]

        # Vérification du chemin racine (train)
        t = test_case()
        if learner_train.root != train_path:
            t.failed = True
            t.msg = f"Chemin racine incorrect pour le train_dataset"
            t.want = f"chemin racine comme 'root={train_path}'"
            t.got = f"chemin racine comme 'root={learner_train.root}'"
        cases.append(t)

        # Vérification du chemin racine (val)
        t = test_case()
        if learner_val.root != val_path:
            t.failed = True
            t.msg = f"Chemin racine incorrect pour le val_dataset"
            t.want = f"chemin racine comme 'root={val_path}'"
            t.got = f"chemin racine comme 'root={learner_val.root}'"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)



def exercise_2_fakefinder_transferlearning(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "define_transformations a un type incorrect"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        mean = torch.tensor([0.4, 0.4, 0.4])
        std = torch.tensor([0.3, 0.3, 0.3])
        expected_function_return = transforms.Compose
        expected_train_rand_resize_crop = (224, 224)
        expected_brightness = 0.2
        expected_contrast = 0.2

        learner_train_transform, learner_val_transform = learner_func(mean=mean, std=std)

        # Vérification du type de retour 1 (train_transform)
        t = test_case()
        if not isinstance(learner_train_transform, expected_function_return):
            t.failed = True
            t.msg = "Type de retour train_transform incorrect"
            t.want = "<class 'torchvision.transforms.transforms.Compose'>"
            t.got = f"{type(learner_train_transform)}"
            return [t]

        # Vérification du type de retour 2 (val_transform)
        t = test_case()
        if not isinstance(learner_val_transform, expected_function_return):
            t.failed = True
            t.msg = "Type de retour val_transform incorrect"
            t.want = "<class 'torchvision.transforms.transforms.Compose'>"
            t.got = f"{type(learner_val_transform)}"
            return [t]

        # Vérifie le nombre de transformations d'entraînement, devrait être 5
        t = test_case()
        if len(learner_train_transform.transforms) != 5:
            t.failed = True
            t.msg = f"5 transformations d'entraînement attendues, mais {len(learner_train_transform.transforms)} trouvées"
            t.want = "5 transformations d'entraînement dans define_transformations. RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor et Normalize"
            t.got = f"{len(learner_train_transform.transforms)} transformations d'entraînement"
            return [t]

        # Vérifie le nombre de transformations de validation, devrait être 3
        t = test_case()
        if len(learner_val_transform.transforms) != 3:
            t.failed = True
            t.msg = f"3 transformations de validation attendues, mais {len(learner_val_transform.transforms)} trouvées"
            t.want = "3 transformations de validation dans define_transformations. Resize, ToTensor et Normalize"
            t.got = f"{len(learner_val_transform.transforms)} transformations de validation"
            return [t]

        ###################################################################################

        # Vérifie RandomResizedCrop dans train_transform
        rand_resized_crop_found = False
        found_correct_size = False
        found_sizes = []
        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.RandomResizedCrop):
                rand_resized_crop_found = True
                found_sizes.append(transform.size)
                if transform.size == expected_train_rand_resize_crop:
                    found_correct_size = True
                    break

        t = test_case()
        if not rand_resized_crop_found:
            t.failed = True
            t.msg = "Transformation RandomResizedCrop non trouvée dans train_transform"
            t.want = "train_transform doit inclure la transformation RandomResizedCrop"
            t.got = "train_transform sans transformation RandomResizedCrop"
        elif not found_correct_size:
            t.failed = True
            t.msg = f"RandomResizedCrop trouvée dans train_transform, mais avec une taille incorrecte"
            t.want = f"{expected_train_rand_resize_crop}"
            t.got = f"{found_sizes[0] if found_sizes else 'Aucune taille incorrecte trouvée'}"
        cases.append(t)

        # Vérifie RandomHorizontalFlip dans train_transform
        hflip_found = False

        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.RandomHorizontalFlip):
                hflip_found = True
                break

        t = test_case()
        if hflip_found == False:
            t.failed = True
            t.msg = "Transformation RandomHorizontalFlip non trouvée dans train_transform"
            t.want = "Transformation RandomHorizontalFlip présente dans train_transform"
            t.got = "Aucune transformation RandomHorizontalFlip dans train_transform"
        cases.append(t)

        # Vérifie ColorJitter dans train_transform avec brightness et contrast spécifiques
        color_jitter_found, found_correct_jitter, found_brightness_val, found_contrast_val = unittests_utils.check_color_jitter(
            learner_train_transform, expected_brightness, expected_contrast
        )

        t = test_case()
        if not color_jitter_found:
            t.failed = True
            t.msg = "Transformation ColorJitter non trouvée dans train_transform"
            t.want = "train_transform doit inclure la transformation ColorJitter"
            t.got = "train_transform sans transformation ColorJitter"
        elif not found_correct_jitter:
            t.failed = True
            t.msg = "ColorJitter trouvée dans train_transform, mais avec brightness et/ou contrast incorrects"
            t.want = f"(brightness={expected_brightness}, contrast={expected_contrast})"
            t.got = f"(brightness={found_brightness_val}, contrast={found_contrast_val})"
        cases.append(t)


        # Vérifie ToTensor dans train_transform
        totensor_found = False

        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.ToTensor):
                totensor_found = True
                break

        t = test_case()
        if totensor_found == False:
            t.failed = True
            t.msg = "Transformation ToTensor non trouvée dans train_transform"
            t.want = "Transformation ToTensor présente dans train_transform"
            t.got = "Aucune transformation ToTensor dans train_transform"
        cases.append(t)


        # Vérifie Normalize dans train_transform avec mean et std spécifiques
        normalize_found = False
        found_correct_normalize = False
        found_mean = None
        found_std = None
        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                normalize_found = True
                found_mean = transform.mean
                found_std = transform.std
                if torch.equal(found_mean, mean) and torch.equal(found_std, std):
                    found_correct_normalize = True
                break

        t = test_case()
        if not normalize_found:
            t.failed = True
            t.msg = "Transformation Normalize non trouvée dans train_transform"
            t.want = "train_transform doit inclure la transformation Normalize"
            t.got = "train_transform sans transformation Normalize"
        elif not found_correct_normalize:
            t.failed = True
            t.msg = "Normalize trouvée dans train_transform, mais avec mean et/ou std incorrects"
            t.want = "(mean=mean, std=std)"
            t.got = f"(mean={tuple(found_mean.tolist()) if found_mean is not None else None}, std={tuple(found_std.tolist()) if found_std is not None else None})"
        cases.append(t)

        ###################################################################################

        expected_val_resize = (224, 224)

        # Vérifie Resize dans val_transform avec une taille spécifique
        resize_found_val = False
        found_correct_resize_val = False
        found_resize_size_val = None
        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.Resize):
                resize_found_val = True
                found_resize_size_val = transform.size
                if transform.size == expected_val_resize:
                    found_correct_resize_val = True
                break

        t = test_case()
        if not resize_found_val:
            t.failed = True
            t.msg = "Transformation Resize non trouvée dans val_transform"
            t.want = "val_transform doit inclure la transformation Resize"
            t.got = "val_transform sans transformation Resize"
        elif not found_correct_resize_val:
            t.failed = True
            t.msg = "Resize trouvée dans val_transform, mais avec une taille de pixel incorrecte"
            t.want = f"{expected_val_resize}"
            t.got = f"{found_resize_size_val}"
        cases.append(t)

        # Vérifie ToTensor dans val_transform
        totensor_found = False

        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.ToTensor):
                totensor_found = True
                break

        t = test_case()
        if totensor_found == False:
            t.failed = True
            t.msg = "Transformation ToTensor non trouvée dans val_transform"
            t.want = "Transformation ToTensor présente dans val_transform"
            t.got = "Aucune transformation ToTensor dans val_transform"
        cases.append(t)

        # Vérifie Normalize dans val_transform avec mean et std spécifiques
        normalize_found = False
        found_correct_normalize = False
        found_mean = None
        found_std = None
        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                normalize_found = True
                found_mean = transform.mean
                found_std = transform.std
                if torch.equal(found_mean, mean) and torch.equal(found_std, std):
                    found_correct_normalize = True
                break

        t = test_case()
        if not normalize_found:
            t.failed = True
            t.msg = "Transformation Normalize non trouvée dans val_transform"
            t.want = "val_transform doit inclure la transformation Normalize"
            t.got = "val_transform sans transformation Normalize"
        elif not found_correct_normalize:
            t.failed = True
            t.msg = "Normalize trouvée dans val_transform, mais avec mean et/ou std incorrects"
            t.want = "(mean=mean, std=std)"
            t.got = f"(mean={tuple(found_mean.tolist()) if found_mean is not None else None}, std={tuple(found_std.tolist()) if found_std is not None else None})"
        cases.append(t)

        ###################################################################################

        # Vérifie l'ordre des transformations (train_transform)
        expected_train_order = [
            transforms.RandomResizedCrop,
            transforms.RandomHorizontalFlip,
            transforms.ColorJitter,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        learner_train_order = [type(transform) for transform in learner_train_transform.transforms]

        t = test_case()
        if learner_train_order != expected_train_order:
            t.failed = True
            t.msg = "Les transformations d'entraînement ne sont pas appliquées dans l'ordre attendu"
            t.want = f"[{', '.join([t.__name__ for t in expected_train_order])}]"
            t.got = f"[{', '.join([t.__name__ for t in learner_train_order])}]"
        cases.append(t)

        # Vérifie l'ordre des transformations (val_transform)
        expected_val_order = [
            transforms.Resize,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        learner_val_order = [type(transform) for transform in learner_val_transform.transforms]

        t = test_case()
        if learner_val_order != expected_val_order:
            t.failed = True
            t.msg = "Les transformations de validation ne sont pas appliquées dans l'ordre attendu"
            t.want = f"[{', '.join([t.__name__ for t in expected_val_order])}]"
            t.got = f"[{', '.join([t.__name__ for t in learner_val_order])}]"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)



def exercise_3_fakefinder_transferlearning(learner_func):
    def g():

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_data_loaders a un type incorrect"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        mock_train, mock_val = unittests_utils.generate_mock_datasets()
        learner_train_loader, learner_val_loader, learner_trainset, learner_valset = learner_func(mock_train, mock_val, batch_size=4)

        # Vérification du type de retour 1 (train_loader)
        t = test_case()
        if not isinstance(learner_train_loader, DataLoader):
            t.failed = True
            t.msg = "train_loader a un type de retour incorrect"
            t.want = DataLoader
            t.got = type(learner_train_loader)
            return [t]

        # Vérification du type de retour 2 (val_loader)
        t = test_case()
        if not isinstance(learner_val_loader, DataLoader):
            t.failed = True
            t.msg = "val_loader a un type de retour incorrect"
            t.want = DataLoader
            t.got = type(learner_val_loader)
            return [t]

        # Vérification du type de retour 3 (train_transform_dataset)
        t = test_case()
        if not isinstance(learner_trainset, (ImageFolder, MockImageFolder)):
            t.failed = True
            t.msg = "trainset a un type de retour incorrect."
            t.want = "Un objet Dataset comme ImageFolder"
            t.got = type(learner_trainset)
            return [t]

        # Vérification du type de retour 4 (val_transform_dataset)
        t = test_case()
        if not isinstance(learner_valset, (ImageFolder, MockImageFolder)):
            t.failed = True
            t.msg = "valset a un type de retour incorrect."
            t.want = "Un objet Dataset comme ImageFolder"
            t.got = type(learner_valset)
            return [t]

        learner_code = inspect.getsource(learner_func)
        cleaned_code = unittests_utils.remove_comments(learner_code)

        # Vérifie la mention de 'define_transformations' dans le code de l'apprenant
        t = test_case()
        if "define_transformations" not in cleaned_code:
            t.failed = True
            t.msg = "L'implémentation de create_data_loaders ne semble pas appeler la fonction 'define_transformations' pour l'initialisation de 'train_transform, val_transform'"
            t.want = "Fonction 'define_transformations' pour l'initialisation de 'train_transform, val_transform'"
            t.got = "'train_transform, val_transform' étant initialisés d'une autre manière"
            return [t]

        # Vérifie si l'attribut transform a été assigné au trainset
        t = test_case()
        if getattr(learner_trainset, 'transform', None) is None:
            t.failed = True
            t.msg = "L'attribut `transform` de l'ensemble d'entraînement retourné est soit manquant soit n'a pas reçu de valeur"
            t.want = "`trainset.transform` doit être assigné aux transformations d'entraînement."
            t.got = "L'attribut `transform` n'a pas été trouvé ou a été défini à `None`."
            return [t]

        # Vérifie si l'attribut transform a été assigné au valset
        t = test_case()
        if getattr(learner_valset, 'transform', None) is None:
            t.failed = True
            t.msg = "L'attribut `transform` de l'ensemble de validation retourné est soit manquant soit n'a pas reçu de valeur"
            t.want = "`valset.transform` doit être assigné aux transformations de validation."
            t.got = "L'attribut `transform` n'a pas été trouvé ou a été défini à `None`."
            return [t]

        # Vérifie si le bon nombre de transformations a été appliqué au trainset
        num_transforms = len(learner_trainset.transform.transforms)

        t = test_case()
        if num_transforms != 5:
            t.failed = True
            t.msg = "Transformations incorrectes appliquées à trainset. Assurez-vous d'appliquer train_transform"
            t.want = "5 transformations attendues à appliquer."
            t.got = f"{num_transforms} transformations trouvées."
        cases.append(t)

        # Vérifie si le bon nombre de transformations a été appliqué au valset
        num_transforms_val = len(learner_valset.transform.transforms)

        t = test_case()
        if num_transforms_val != 3:
            t.failed = True
            t.msg = "Transformations incorrectes appliquées à valset. Assurez-vous d'appliquer val_transform"
            t.want = "3 transformations attendues à appliquer."
            t.got = f"{num_transforms_val} transformations trouvées."
        cases.append(t)

        expected_train_shape = torch.Size([4, 3, 224, 224])
        expected_val_shape = torch.Size([4, 3, 224, 224])

        ### Check train_loader
        for batch_idx, (images, labels) in enumerate(learner_train_loader):
            if batch_idx == 2:
                learner_train_shape = images.shape
                break

        # Vérifie l'initialisation correcte de train_loader avec train_transform_dataset
        t = test_case()
        if len(learner_train_loader) != 25:
            t.failed = True
            t.msg = "Longueur incorrecte de train_loader. Assurez-vous d'utiliser train_transform_dataset lors de la configuration de train_loader"
            t.want = "train_loader doit utiliser train_transform_dataset"
            t.got = "train_loader n'utilise pas train_transform_dataset"
        cases.append(t)

        # Taille de batch (train_loader)
        t = test_case()
        if expected_train_shape[0] != learner_train_shape[0]:
            t.failed = True
            t.msg = "batch_size incorrect de train_loader"
            t.want = "batch_size=batch_size"
            t.got = f"batch_size={learner_train_shape[0]}"
        cases.append(t)

        # Vérifie val_loader
        for batch_idx, (images, labels) in enumerate(learner_val_loader):
            if batch_idx == 2:
                learner_val_shape = images.shape
                break

        # Vérifie l'initialisation correcte de val_loader avec val_transform_dataset
        t = test_case()
        if len(learner_val_loader) != 19:
            t.failed = True
            t.msg = "Longueur incorrecte de val_loader. Assurez-vous d'utiliser val_transform_dataset lors de la configuration de val_loader"
            t.want = "val_loader doit utiliser val_transform_dataset"
            t.got = "val_loader n'utilise pas val_transform_dataset"
        cases.append(t)

        # Taille de batch (val_loader)
        t = test_case()
        if expected_val_shape[0] != learner_val_shape[0]:
            t.failed = True
            t.msg = "batch_size incorrect de val_loader"
            t.want = "batch_size=batch_size"
            t.got = f"batch_size={learner_val_shape[0]}"
        cases.append(t)

        # Vérifie shuffle (train_loader)
        t = test_case()
        if not unittests_utils.check_shuffle(learner_train_loader, True):
            t.failed = True
            t.msg = "shuffle incorrect de train_loader. Assurez-vous de définir shuffle comme True pour le train_loader"
            t.want = "shuffle=True"
            t.got = "shuffle=False ou shuffle=None"
        cases.append(t)

        # Vérifie shuffle (val_loader)
        t = test_case()
        if not unittests_utils.check_shuffle(learner_val_loader, False):
            t.failed = True
            t.msg = "shuffle incorrect de val_loader. Assurez-vous de définir shuffle comme False pour le val_loader"
            t.want = "shuffle=False"
            t.got = "shuffle=True ou shuffle=None"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)



def exercise_4_1_fakefinder_transferlearning(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "load_mobilenetv3_model a un type incorrect"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        local_weights = "./models/mobilenet_v3_large-8738ca79.pth"
        learner_model = learner_func(local_weights)

        # Vérification du type de retour
        t = test_case()
        expected_type = MobileNetV3
        if not isinstance(learner_model, expected_type):
            t.failed = True
            t.msg = f"Type de modèle retourné incorrect. Assurez-vous de charger le modèle 'mobilenet_v3_large' comme demandé"
            t.want = f"modèle de type {expected_type}"
            t.got = f"{type(learner_model)}"
            return [t]

        # Vérifie la dernière couche du classificateur
        t = test_case()
        try:
            last_classifier = learner_model.classifier[-1]
            is_linear = isinstance(last_classifier, nn.Linear)
            has_correct_in_features = last_classifier.in_features == 1280
            has_correct_out_features = last_classifier.out_features == 1000

            if not (is_linear and has_correct_in_features and has_correct_out_features):
                t.failed = True
                t.msg = "La dernière couche du modèle du classificateur ne correspond pas à la couche Linear attendue (in_features=1280, out_features=1000)."
                t.want = "Modèle chargé doit être 'mobilenet_v3_large' comme demandé"
                t.got = "Un autre modèle que 'mobilenet_v3_large' est chargé"
                return [t]
        except Exception as e:
            t.failed = True
            t.msg = f"Une erreur s'est produite lors de l'accès ou de la vérification de la dernière couche du classificateur: {e}"
            t.want = "Modèle chargé doit être 'mobilenet_v3_large' comme demandé"
            t.got = "Un autre modèle que 'mobilenet_v3_large' est chargé"
            return [t]

        # Vérifie la présence de 'weights=None'
        learner_code = inspect.getsource(learner_func)
        cleaned_code = unittests_utils.remove_comments(learner_code)
        pretrained_flag_pattern = r"weights\s*=\s*None"
        t = test_case()
        if not re.search(pretrained_flag_pattern, cleaned_code):
            t.failed = True
            t.msg = "Le paramètre 'weights' ne semble pas être défini comme None"
            t.want = "weights=None"
            t.got = "Autre chose"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)



def exercise_4_2_fakefinder_transferlearning(learner_func):
    def g():

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "update_model_last_layer a un type incorrect"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        initial_model = tv_models.mobilenet_v3_large(weights=None)
        local_weights = "./models/mobilenet_v3_large-8738ca79.pth"
        state_dict = torch.load(local_weights, map_location=torch.device('cpu'))
        initial_model.load_state_dict(state_dict)
        learner_output = learner_func(initial_model, num_classes=8)

        # Vérification du type de retour
        t = test_case()
        expected_type = MobileNetV3
        if not isinstance(learner_output, expected_type):
            t.failed = True
            t.msg = f"Type de modèle retourné incorrect"
            t.want = f"{expected_type}"
            t.got = f"{type(learner_output)}"
            return [t]

        # Vérifie si les paramètres de features sont gelés
        t = test_case()
        all_frozen = True
        for name, param in learner_output.features.named_parameters():
            if param.requires_grad:
                all_frozen = False
                break
        if not all_frozen:
            t.failed = True
            t.msg = "Tous les paramètres de features ne sont pas gelés (requires_grad n'est pas False)"
            t.want = "'requires_grad = False' pour toutes les couches de features du modèle"
            t.got = "'requires_grad' n'est pas false pour une ou plusieurs couches de features du modèle"
        cases.append(t)

        # Vérifie si la dernière couche du classificateur est une nouvelle couche nn.Linear
        t = test_case()
        if not isinstance(learner_output.classifier[-1], nn.Linear):
            t.failed = True
            t.msg = "La dernière couche du classificateur devrait être une nouvelle couche nn.Linear."
            t.want = nn.Linear
            t.got = type(learner_output.classifier[-1])
            return [t]

        # Vérifie les features de sortie de la dernière couche linéaire
        t = test_case()
        last_layer = learner_output.classifier[-1]
        expected_out_features = 8
        if not hasattr(last_layer, 'out_features') or last_layer.out_features != expected_out_features:
            t.failed = True
            t.msg = f"La dernière couche linéaire ne correspond pas aux features de sortie attendues"
            if hasattr(last_layer, 'in_features') and last_layer.in_features == 1280:
                t.want = "Linear(in_features=1280, out_features=num_classes, bias=True)"
                t.got = f"{last_layer}"
            else:
                t.want = "Linear(in_features=num_features, out_features=num_classes, bias=True)"
                t.got = f"{last_layer}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)