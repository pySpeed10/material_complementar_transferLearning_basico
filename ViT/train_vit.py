import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import timm  # Importante: instale com `pip install timm`

# Reprodutibilidade
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
batch_size = 16
num_epochs = 60
num_folds = 5
save_path = "resultados_vit_6" # 1 => lr = 0.001  2 => lr = 0.01  3 => lr = 0.1 
                               #4 => lr = 0.001  5 => lr = 0.01  6 => lr = 0.1 
os.makedirs(save_path, exist_ok=True)

# Transforms
data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carregamento do dataset
dataset = torchvision.datasets.ImageFolder(
    root='Mini_DDSM_Upload',
    transform=data_transforms
)
targets = np.array([sample[1] for sample in dataset])
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

# Armazenamento de métricas globais
y_true_all, y_pred_all = [], []
all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), targets)):
    print(f"Fold {fold + 1}/{num_folds}")
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Carregar Vision Transformer pré-treinado
    model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=num_classes) #vit base = vit_base_patch16_224;
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # taxa menor para estabilidade

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / total)
        train_accuracies.append(correct / total)

        # Validação
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_losses.append(running_loss / total)
        val_accuracies.append(correct / total)

    # Avaliação final do fold
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    specificity = np.mean(specificities)

    all_metrics.append((acc, recall, precision, f1, specificity))

    # Salvar gráficos
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de Confusão - Fold {fold + 1}")
    plt.savefig(os.path.join(save_path, f"matriz_confusao_fold{fold + 1}.png"))
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title(f"Acurácia - Fold {fold + 1}")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"acuracia_fold{fold + 1}.png"))
    plt.close()

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"Loss - Fold {fold + 1}")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"loss_fold{fold + 1}.png"))
    plt.close()

# Métricas finais combinadas
final_acc = accuracy_score(y_true_all, y_pred_all)
final_recall = recall_score(y_true_all, y_pred_all, average='macro')
final_precision = precision_score(y_true_all, y_pred_all, average='macro')
final_f1 = f1_score(y_true_all, y_pred_all, average='macro')
cm = confusion_matrix(y_true_all, y_pred_all)
specificities = []
for i in range(num_classes):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)
final_specificity = np.mean(specificities)

# Matriz de confusão final
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão Final")
plt.savefig(os.path.join(save_path, "matriz_confusao_final.png"))
plt.close()

# Salvar métricas
with open(os.path.join(save_path, "metricas_finais.txt"), "w") as f:
    for i, (acc, rec, prec, f1_, spec) in enumerate(all_metrics):
        f.write(f"Fold {i + 1} - Acc: {acc:.4f}, Recall: {rec:.4f}, Prec: {prec:.4f}, F1: {f1_:.4f}, Esp: {spec:.4f}\n")
    f.write("\n")
    f.write(f"FINAL - Acc: {final_acc:.4f}, Recall: {final_recall:.4f}, Prec: {final_precision:.4f}, F1: {final_f1:.4f}, Esp: {final_specificity:.4f}\n")

# Salvar modelo final
torch.save(model.state_dict(), os.path.join(save_path, "modelo_vit_final.pth"))