import os
import torch
import torchmetrics
from tqdm.auto import tqdm


def training_loop_with_best_model(model, train_loader, val_loader, loss_fcn, optmzr, device, num_epochs=3, scheduler=None):
    """
    Exécute la boucle d'entraînement et de validation pour un modèle PyTorch donné.
    Sauvegarde et retourne le modèle avec la meilleure précision de validation.

    Args:
        model: Le modèle à entraîner.
        train_loader: Le data loader pour le dataset d'entraînement.
        val_loader: Le data loader pour le dataset de validation.
        loss_fcn: La fonction de perte pour calculer la perte d'entraînement.
        optmzr: L'optimiseur pour mettre à jour les paramètres du modèle.
        device: Le périphérique (CPU ou CUDA) sur lequel le modèle et les données seront traités.
        num_epochs: Le nombre total d'époques pour l'entraînement.

    Returns:
        L'objet modèle entraîné avec les poids qui ont atteint la meilleure précision de validation.
    """
    # Crée le répertoire pour sauvegarder le meilleur modèle s'il n'existe pas.
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model_nanobanana_pro.pth")

    # Déplace le modèle vers le périphérique de calcul spécifié.
    model.to(device)
    # Assigne la fonction de perte fournie.
    loss_function = loss_fcn
    # Assigne l'optimiseur fourni.
    optimizer = optmzr

    # Détermine le nombre de classes depuis la couche de sortie finale du modèle.
    num_classes = model.classifier[-1].out_features

    # Initialise les métriques de précision, precision et recall pour la validation.
    val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)

    # Initialise les variables pour suivre les meilleures performances.
    best_val_accuracy = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0

    # Liste pour stocker les métriques par époque
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Commence la boucle principale d'entraînement et de validation pour un nombre d'époques.
    for epoch in range(num_epochs):
        # Met le modèle en mode entraînement.
        model.train()
        # Initialise une variable pour accumuler la perte d'entraînement.
        running_loss = 0.0
        # Initialise un compteur pour les échantillons d'entraînement correctement classés.
        total_train_correct = 0
        # Initialise un compteur pour le total d'échantillons d'entraînement.
        total_train_samples = 0

        # Crée une barre de progression pour les batchs d'entraînement.
        train_progress_bar = tqdm(train_loader, desc=f"Époque {epoch + 1}/{num_epochs} Entraînement", unit="batch")
        # Itère sur les batchs depuis le data loader d'entraînement.
        for images, labels in train_progress_bar:
            # Déplace les images et labels vers le périphérique désigné.
            images, labels = images.to(device), labels.to(device)
            # Réinitialise les gradients de tous les tenseurs optimisés à zéro.
            optimizer.zero_grad()
            # Effectue une passe forward pour obtenir les sorties du modèle.
            outputs = model(images)
            # Calcule la perte entre les sorties et les vrais labels.
            loss = loss_function(outputs, labels)
            # Effectue la rétropropagation pour calculer les gradients.
            loss.backward()
            # Met à jour les poids du modèle en utilisant l'optimiseur.
            optimizer.step()

            # Accumule la perte et les compteurs d'échantillons.
            running_loss += loss.item() * labels.size(0)
            # Obtient la classe prédite avec la probabilité la plus élevée.
            _, predicted = torch.max(outputs, dim=1)
            # Met à jour le compteur pour les prédictions correctes.
            total_train_correct += (predicted == labels).sum().item()
            # Met à jour le compte total d'échantillons traités.
            total_train_samples += labels.size(0)

            # Calcule la perte moyenne pour l'époque actuelle.
            epoch_loss = running_loss / total_train_samples
            # Calcule la précision pour l'époque actuelle.
            epoch_acc = 100 * total_train_correct / total_train_samples
            # Met à jour la barre de progression avec la perte et la précision en temps réel.
            train_progress_bar.set_postfix(loss=f"{epoch_loss:.4f}", accuracy=f"{epoch_acc:.2f}%")

        # Commence la phase de validation.
        # Met le modèle en mode évaluation.
        model.eval()
        # Initialise un compteur pour le total d'échantillons de validation.
        total_val_samples = 0
        # Initialise une variable pour accumuler la perte de validation.
        val_loss = 0.0

        # Réinitialise les objets de métriques de validation pour la nouvelle époque.
        val_accuracy_metric.reset()
        val_precision_metric.reset()
        val_recall_metric.reset()

        # Désactive les calculs de gradient pour l'efficacité pendant la validation.
        with torch.no_grad():
            # Crée une barre de progression pour les batchs de validation.
            val_progress_bar = tqdm(val_loader, desc=f"Époque {epoch + 1}/{num_epochs} Validation", unit="batch")
            # Itère sur les batchs depuis le data loader de validation.
            for images, labels in val_progress_bar:
                # Déplace les images et labels vers le périphérique désigné.
                images, labels = images.to(device), labels.to(device)
                # Effectue une passe forward.
                outputs = model(images)
                # Calcule la perte.
                loss = loss_function(outputs, labels)
                # Accumule la perte de validation.
                val_loss += loss.item() * labels.size(0)
                # Obtient la classe prédite.
                _, predicted = torch.max(outputs, dim=1)
                # Met à jour le compte total d'échantillons de validation.
                total_val_samples += labels.size(0)

                # Met à jour les objets de métriques avec les prédictions et labels du batch.
                val_accuracy_metric.update(predicted, labels)
                val_precision_metric.update(predicted, labels)
                val_recall_metric.update(predicted, labels)

                # Met à jour la barre de progression avec la précision actuelle.
                val_progress_bar.set_postfix(
                    accuracy=f"{100 * val_accuracy_metric.compute():.2f}%"
                )

        # Calcule la perte de validation moyenne pour l'époque.
        avg_val_loss = val_loss / total_val_samples

        # Calcule la perte d'entraînement moyenne pour l'époque.
        avg_train_loss = running_loss / total_train_samples
        # Calcule la précision d'entraînement pour l'époque.
        train_acc = total_train_correct / total_train_samples

        # Calcule les valeurs de métriques finales pour l'époque entière.
        final_val_acc = val_accuracy_metric.compute()
        final_val_precision = val_precision_metric.compute()
        final_val_recall = val_recall_metric.compute()

        # Stocke les métriques pour cette époque
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(final_val_acc.item())

        # Affiche un résumé des résultats de validation pour l'époque.
        print(f'Perte Val (Moy): {avg_val_loss:.4f}, Précision Val: {final_val_acc * 100:.2f}%\n')

        # Vérifie si le modèle actuel est le meilleur et le sauvegarde.
        if final_val_acc > best_val_accuracy:
            best_val_accuracy = final_val_acc
            best_val_precision = final_val_precision
            best_val_recall = final_val_recall
            torch.save(model.state_dict(), best_model_path)
            print(f"Nouveau meilleur modèle sauvegardé dans {best_model_path} avec Précision Val: {best_val_accuracy * 100:.2f}%\n")

        # Met à jour le scheduler si fourni
        if scheduler is not None:
            scheduler.step()

    # Affiche un message indiquant la fin de l'entraînement.
    print("\nEntraînement terminé. Meilleur modèle entraîné retourné.")
    print(f"Meilleure Exactitude Val: {best_val_accuracy * 100:.2f}%")
    print(f"Meilleure Precision Val: {best_val_precision:.4f}")
    print(f"Meilleur Recall Val: {best_val_recall:.4f}\n")

    # Charge les poids du meilleur modèle avant de le retourner.
    model.load_state_dict(torch.load(best_model_path))

    # Retourne le meilleur modèle entraîné et les métriques.
    metrics = (train_losses, train_accuracies, val_losses, val_accuracies)
    return model, metrics
