# Fake Finder Nano Banana Pro

Détecteur d'images générées par IA utilisant **MobileNetV3 Large** finetuné avec des images Nano Banana Pro pour identifier les fakes.

**Précision : 90% (9 images sur 10)**
(Fonctionne aussi sur images difussion - Midjourney, SD, DALL-E)

*Note : les datasets d'images d'entrainement sont à télécharger sur HuggingFace 👇*

![RAG Agentique multi-agent Header](./images/fake-1.png)

## 🎯 Concept Principal : Finetuning par Transfer Learning

Ce projet utilise la technique du **transfer learning en changeant seulement la dernière couche - la couche classfieur**:

1. **ImageNet → Fake général** : Fine-tuning sur SD/Midjourney/DALL-E
2. **Fake général → Nano Banana Pro** : Fine-tuning spécifique sur Nano Banana Pro

## 🔄 Transfer Learning

Ce projet repose entièrement sur une stratégie de **transfer learning** en cascade :

### Étape 1 : Transfer Learning vers la détection fake/real
- **Source** : Modèle ImageNet v3 Large (`mobilenet_v3_large-8738ca79.pth - même version que dans le doc PyTorch`)
- **Cible** : Détection générale d'images fake (SD, Midjourney, DALL-E)
- **Méthode** : Fine-tuning du classifier (features extractor gelé)
- **Résultat** : `best_model_midjourney_dalle_sd.pth`

### Étape 2 : Transfer Learning vers Nano Banana Pro
- **Source** : Modèle fine-tuné SD/Midjourney/DALL-E
- **Cible** : Détection spécifique Nano Banana Pro
- **Méthode** : Fine-tuning du classifier avec learning rate réduit (0.0005)
- **Résultat** : `best_model_nanobanana_pro.pth`

**Avantages du transfer learning** :
- ✅ Réutilisation des connaissances pré-existantes
- ✅ Entraînement rapide avec peu de données **(1 seule Epoch)**
- ✅ Meilleures performances que l'entraînement from scratch
- ✅ Adaptation progressive du modèle général vers le cas spécifique

## 🏗️ Architecture

- **Modèle de base** : MobileNetV3-Large (transfer learning depuis ImageNet)
- **Fine-tuning par Transfer learning** : Cascade en 3 étapes (ImageNet → Fake général midjourney/dall-e/SD → Puis Nano Banana Pro)
- **Fine-tuning** : couche classifier uniquement (features extractor gelé)
- **Classes** : 2 (Real / Fake)

## 🚀 Installation

```bash
# Installation des dépendances
uv sync

# Téléchargement des datasets depuis Hugging Face
uv run python download_dataset_images.py julienlucas/midjourney-dalle-sd-dataset ./AIvsReal_midjourney_dalle_sd
uv run python download_dataset_images.py julienlucas/nanobanana-pro-dataset ./AIvsReal_nanobanana_pro
```

## 🎓 Entraînement (Transfer Learning)

### 1. Fine-tuning initial (SD/Midjourney/DALL-E)

```bash
uv run python finetune_midjourney_dalle_sd.py
```

Génère `models/best_model_midjourney_dalle_sd.pth`

### 2. Puis fine-tuning Nano Banana Pro

```bash
uv run python finetune_nanobananapro.py
```

Génère `models/best_model_nanobanana_pro.pth`

## 🔍 Inférence

```bash
# Inférence simple avec Grad-CAM
uv run python inference.py

# Évaluation complète du dataset de test
uv run python inference_check_test_dataset.py
```

## 📊 Résultats

| Dataset | Modèle | Accuracy |
|---------|--------|----------|
| Midjourney/DALL-E/SD | `best_model_midjourney_dalle_sd.pth` | 83.40% |
| Nano Banana Pro (après fine-tuning) | `best_model_nanobanana_pro.pth` | 89.40% |

## 📄 Licence

Ce projet est destiné à l'éducation sur l'IA sur Youtube: https://www.youtube.com/@julienlucas

Mettez une ⭐ pour soutenir mon travail 🙏