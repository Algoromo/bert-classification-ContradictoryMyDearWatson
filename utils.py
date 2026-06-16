import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def get_device():
    """
    Returns:
        torch.device: Le device à utiliser pour l'entraînement/l'inférence,  (cuda si disponible, sinon cpu)
    """ 

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    """
    Args:
        seed (int): Valeur de la seed à utiliser pour random, numpy et torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, labels):
    """
    Calcule l'accuracy et le F1-score

    Args:
        preds: Labels prédits par le modèle
        labels: Labels réels (ground truth)

    Returns:
        tuple[float, float]: (accuracy, f1_macro)
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


def plot_training_curves(history, save_path="images\\training_curves.png"):
    """ 
    Trace et sauvegarde les courbes loss/accuracy train vs validation

    Args:
        history (dict): Dictionnaire contenant les listes 'train_loss','val_loss', 'train_accuracy', 'val_accuracy' par epoch
        save_path (str): Chemin du fichier image à sauvegarder.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Évolution de la Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", marker="o")
    axes[1].set_title("Évolution de l'Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="./images/confusion_matrix.png"):
    """
    Trace et sauvegarde la matrice de confusion.

    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        class_names (list[str]): Noms des classes pour les axes
        save_path (str): Chemin du fichier image à sauvegarder
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")
    ax.set_title("Matrice de confusion")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()