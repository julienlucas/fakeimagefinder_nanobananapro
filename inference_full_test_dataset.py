import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_mobilenetv3_model(weights_path, num_classes=None):
    model = tv_models.mobilenet_v3_large(weights=None)
    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


trained_model_path = "./models/best_model_nanobanana_pro.pth"
model = load_mobilenetv3_model(trained_model_path, num_classes=2)
model = model.to(DEVICE)
model.eval()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

REAL_THRESHOLD = 0.7
FAKE_THRESHOLD = 0.7

base_dir = "./AIvsReal_nanobanana_pro/test"
real_dir = os.path.join(base_dir, "real")
fake_dir = os.path.join(base_dir, "fake")

real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
fake_images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

print(f"📁 Test REAL: {len(real_images)} images")
print(f"📁 Test FAKE: {len(fake_images)} images")
print("=" * 80)

real_true_positives = 0
real_false_negatives = 0
real_confidences = []

fake_true_positives = 0
fake_false_negatives = 0
fake_confidences = []

print("\n🔍 Test des images REAL:")
print("-" * 80)
for img_name in real_images:
    img_path = os.path.join(real_dir, img_name)
    pil_image = Image.open(img_path).convert("RGB")
    input_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        real_conf = float(probs[1].item())
        fake_conf = float(probs[0].item())

    if real_conf >= REAL_THRESHOLD:
        predicted = "REAL"
        real_true_positives += 1
        real_confidences.append(real_conf)
    elif fake_conf >= FAKE_THRESHOLD:
        predicted = "FAKE"
        real_false_negatives += 1
    else:
        predicted = "REAL" if real_conf > fake_conf else "FAKE"
        if predicted == "REAL":
            real_true_positives += 1
            real_confidences.append(real_conf)
        else:
            real_false_negatives += 1

    status = "✅" if predicted == "REAL" else "❌"
    print(f"{img_name:40s} | {status} {predicted:4s} | Real: {real_conf*100:5.1f}% | Fake: {fake_conf*100:5.1f}%")

print("\n🔍 Test des images FAKE:")
print("-" * 80)
for img_name in fake_images:
    img_path = os.path.join(fake_dir, img_name)
    pil_image = Image.open(img_path).convert("RGB")
    input_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        real_conf = float(probs[1].item())
        fake_conf = float(probs[0].item())

    if fake_conf >= FAKE_THRESHOLD:
        predicted = "FAKE"
        fake_true_positives += 1
        fake_confidences.append(fake_conf)
    elif real_conf >= REAL_THRESHOLD:
        predicted = "REAL"
        fake_false_negatives += 1
    else:
        predicted = "FAKE" if fake_conf > real_conf else "REAL"
        if predicted == "FAKE":
            fake_true_positives += 1
            fake_confidences.append(fake_conf)
        else:
            fake_false_negatives += 1

    status = "✅" if predicted == "FAKE" else "❌"
    print(f"{img_name:40s} | {status} {predicted:4s} | Real: {real_conf*100:5.1f}% | Fake: {fake_conf*100:5.1f}%")

print("\n" + "=" * 80)
print("📊 STATISTIQUES COMPLÈTES")
print("=" * 80)

total_real = len(real_images)
total_fake = len(fake_images)
total_images = total_real + total_fake

real_precision = real_true_positives / (real_true_positives + fake_false_negatives) if (real_true_positives + fake_false_negatives) > 0 else 0
real_recall = real_true_positives / total_real if total_real > 0 else 0
real_f1 = 2 * (real_precision * real_recall) / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0

fake_precision = fake_true_positives / (fake_true_positives + real_false_negatives) if (fake_true_positives + real_false_negatives) > 0 else 0
fake_recall = fake_true_positives / total_fake if total_fake > 0 else 0
fake_f1 = 2 * (fake_precision * fake_recall) / (fake_precision + fake_recall) if (fake_precision + fake_recall) > 0 else 0

accuracy = (real_true_positives + fake_true_positives) / total_images if total_images > 0 else 0

print(f"\n🎯 Classe REAL:")
print(f"   Vrais positifs (TP): {real_true_positives}/{total_real}")
print(f"   Faux négatifs (FN): {real_false_negatives}/{total_real}")
print(f"   Précision: {real_precision*100:.2f}%")
print(f"   Recall: {real_recall*100:.2f}%")
print(f"   F1-Score: {real_f1*100:.2f}%")
if real_confidences:
    avg_conf = sum(real_confidences) / len(real_confidences)
    print(f"   Confiance moyenne (REAL corrects): {avg_conf*100:.1f}%")

print(f"\n🎯 Classe FAKE:")
print(f"   Vrais positifs (TP): {fake_true_positives}/{total_fake}")
print(f"   Faux négatifs (FN): {fake_false_negatives}/{total_fake}")
print(f"   Précision: {fake_precision*100:.2f}%")
print(f"   Recall: {fake_recall*100:.2f}%")
print(f"   F1-Score: {fake_f1*100:.2f}%")
if fake_confidences:
    avg_conf = sum(fake_confidences) / len(fake_confidences)
    print(f"   Confiance moyenne (FAKE corrects): {avg_conf*100:.1f}%")

print(f"\n📈 MÉTRIQUES GLOBALES:")
print(f"   Accuracy globale: {accuracy*100:.2f}%")
print(f"   Total images testées: {total_images}")
print(f"   Correctement classées: {real_true_positives + fake_true_positives}/{total_images}")
print(f"   Erreurs: {real_false_negatives + fake_false_negatives}/{total_images}")

print("\n" + "=" * 80)
