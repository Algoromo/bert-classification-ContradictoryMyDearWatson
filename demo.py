# interface Gradio


"""
Interface de démonstration interactive avec Gradio.

Charge le meilleur modèle sauvegardé (checkpoints/best_model) et permet
à l'utilisateur de saisir une prémisse et une hypothèse pour obtenir
la relation prédite (entailment / neutral / contradiction) avec les
probabilités associées.

"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "checkpoints/best_model"
CLASS_NAMES = ["entailment", "neutral", "contradiction"]
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("Modèle chargé.")


def predict(premise: str, hypothesis: str):
    """Prédit la relation logique entre une prémisse et une hypothèse.

    Args:
        premise (str): Texte de la prémisse.
        hypothesis (str): Texte de l'hypothèse.

    Returns:
        dict[str, float]: Dictionnaire {classe: probabilité} affichable
            par le composant gr.Label de Gradio.
    """
    if not premise.strip() or not hypothesis.strip():
        return {name: 0.0 for name in CLASS_NAMES}

    encoding = tokenizer(
        premise,
        hypothesis,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

    return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}


examples = [
    [
        "A man is playing guitar on a stage in front of a large crowd.",
        "The man is performing music for an audience.",
    ],
    [
        "Le chat dort paisiblement sur le canapé du salon.",
        "Le chat court dans le jardin.",
    ],
]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Prémisse (Premise)", placeholder="Saisissez la phrase de référence...", lines=3),
        gr.Textbox(label="Hypothèse (Hypothesis)", placeholder="Saisissez l'hypothèse à évaluer...", lines=2),
    ],
    outputs=gr.Label(label="Probabilités par classe", num_top_classes=3),
    title="Classification NLI multilingue (BERT fine-tuné)",
    description=(
        "Cette démo utilise un modèle BERT multilingue (bert-base-multilingual-cased) "
        "fine-tuné sur le dataset 'Contradictory, My Dear Watson'. "
        "Donnez une prémisse et une hypothèse (dans n'importe quelle langue parmi les 15 "
        "supportées) pour prédire la relation logique : entailment, neutral ou contradiction."
    ),
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()