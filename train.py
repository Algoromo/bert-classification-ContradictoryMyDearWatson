# boucles train_epoch / eval_epoch + main

"""
Script principal d'entraînement : charge les données, fine-tune BERT
(bert-base-multilingual-cased) avec une boucle PyTorch manuelle
(pas de Trainer Hugging Face), évalue sur un split de validation
stratifié 80/20, et sauvegarde le meilleur modèle (best val_loss).

"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import NLIDataset
from model import load_tokenizer, load_model, MODEL_NAME, NUM_LABELS
from utils import set_seed,get_device, compute_metrics, plot_training_curves, plot_confusion_matrix

# ----------------------------------------------------------------------
# Hyperparamètres
# ----------------------------------------------------------------------
DATA_PATH = "data/train 6.csv"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model")

MAX_LENGTH = 128       # ~99e percentile de la longueur combinée premise+hypothesis
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-5    # 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

CLASS_NAMES = ["entailment", "neutral", "contradiction"]


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Exécute une époque d'entraînement sur l'ensemble du dataloader.

    Args:
        model: Modèle BERT à entraîner.
        dataloader (DataLoader): Dataloader d'entraînement.
        optimizer: Optimiseur (AdamW).
        scheduler: Scheduler de learning rate.
        device (torch.device): Device sur lequel exécuter le calcul.

    Returns:
        tuple[float, float]: (loss moyenne, accuracy) sur l'epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc="Train", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy, _ = compute_metrics(all_preds, all_labels)
    return avg_loss, accuracy


def eval_epoch(model, dataloader, device):
    """Évalue le modèle sur le dataloader de validation (sans calcul de gradients).

    Args:
        model: Modèle BERT à évaluer.
        dataloader (DataLoader): Dataloader de validation.
        device (torch.device): Device sur lequel exécuter le calcul.

    Returns:
        tuple[float, float, float, list, list]:
            (loss moyenne, accuracy, f1_macro, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc="Eval", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy, f1 = compute_metrics(all_preds, all_labels)
    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    """Pipeline complet : chargement des données, entraînement, sauvegarde."""
    set_seed(SEED)
    device = get_device()
    print(f"Device utilisé : {device}")

    # ------------------------------------------------------------------
    # Chargement et inspection du dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Nombre total d'exemples : {len(df)}")
    print(f"Nombre de classes : {df['label'].nunique()}")
    print("Distribution des classes :")
    print(df["label"].value_counts())
    print("\nExemples :")
    print(df[["premise", "hypothesis", "label"]].head(5))

    # ------------------------------------------------------------------
    # Split train / validation 80/20 stratifié
    # ------------------------------------------------------------------
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    print(f"\nTrain : {len(train_df)} | Validation : {len(val_df)}")

    # ------------------------------------------------------------------
    # Tokenizer, Datasets, DataLoaders
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(MODEL_NAME)

    train_dataset = NLIDataset(
        premises=train_df["premise"].tolist(),
        hypotheses=train_df["hypothesis"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    val_dataset = NLIDataset(
        premises=val_df["premise"].tolist(),
        hypotheses=val_df["hypothesis"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # Modèle, optimiseur, scheduler
    # ------------------------------------------------------------------
    model = load_model(MODEL_NAME, NUM_LABELS).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ------------------------------------------------------------------
    # Boucle d'entraînement
    # ------------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "val_f1_score": [], "learning_rate": [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_f1_score"].append(val_f1)
        history["learning_rate"].append(current_lr)

        # Sauvegarde du meilleur modèle (best val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Nouveau meilleur modèle (val_loss={val_loss:.4f}), sauvegarde...")
            model.save_pretrained(BEST_MODEL_PATH)
            tokenizer.save_pretrained(BEST_MODEL_PATH)

            # Sauvegarde la matrice de confusion du meilleur epoch
            plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES, save_path="./images/confusion_matrix.png")

    # ------------------------------------------------------------------
    # Visualisation des courbes d'apprentissage
    # ------------------------------------------------------------------
    plot_training_curves(history, save_path="./images/training_curves.png")
    print("\nEntraînement terminé.")
    print(f"Meilleur modèle sauvegardé dans : {BEST_MODEL_PATH}")
    print("Courbes : training_curves.png | Matrice de confusion : confusion_matrix.png")


if __name__ == "__main__":
    main()