# Dataset loding
df = pd.read_csv("data\train 6.csv")

# EDA

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}\n")

print("Aperçu")
display(df.head())

print("\n")
print(df.info())

## Répartion des classes
print("\n Labels distribution:")
label_counts = df["label"].value_counts()
label_probs = df["label"].value_counts(normalize=True) * 100
for lbl, count, prob in zip(
    label_counts.index, label_counts.values, label_probs.values
):
    print(f"Label {lbl} : {count} ({prob:.2f}%)")

## Distribution des langues
print("\n Distribution des langues")
print(df["language"].value_counts().head(10))


## Analyse de la longueur du texte
df["premise_len"] = df["premise"].apply(lambda x: len(str(x).split()))
df["hypothesis_len"] = df["hypothesis"].apply(lambda x: len(str(x).split()))

display(df[["premise_len", "hypothesis_len"]].describe())

#VISUALIZATIONS

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Graphique des labels
sns.countplot(x="label", data=df, ax=axes[0], palette="viridis")
axes[0].set_title("Distribution des Labels")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Nombre d'exemples")

# Distribution des longueurs
sns.kdeplot(df["premise_len"], ax=axes[1], label="Prémisse", fill=True)
sns.kdeplot(df["hypothesis_len"], ax=axes[1], label="Hypothèse", fill=True)
axes[1].set_title("Distribution de la longueur des textes (mots)")
axes[1].set_xlabel("Nombre de mots")
axes[1].legend()

plt.tight_layout()
plt.show()
