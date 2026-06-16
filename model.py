import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification)

MODEL_NAME = "bert-base-multilingual-cased"
NUM_LABELS = 3 

# Charge le tokenizer pré-entraîné correspondant au modèle.
def load_tokenizer(model_name: str = MODEL_NAME):
    """
    Returns:
        PreTrainedTokenizer: Le tokenizer chargé
    """
    return AutoTokenizer.from_pretrained(model_name)


# Charge le modèle BERT pré-entraîné avec une tête de classification.
def load_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):

    """
    Args:
        model_name (str): Nom du modèle
        num_labels (int): Nombre de classes de sortie

    Returns:
        PreTrainedModel: Le modèle prêt pour le fine-tuning.
    """
  
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    #AutoModelForSequenceClassification ajoute automatiquement une couche
    #linéaire de classification (dropout + dense) sur la représentation
    #du token [CLS], dimensionnée selon num_labels.

    return model
