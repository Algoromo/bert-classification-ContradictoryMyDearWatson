# Fine-Tuning de BERT pour la Classification de Texte (NLI Multilingue)


**Membres du binôme :** Adji Fatou Mahmoud Ibrahima MBAYE & Michel KHAN 
**Module :** Deep Learning (M2 IA)

---

## 1. Présentation du dataset & Analyse Exploratoire (EDA)

Le sujet choisi porte sur un problème d'implication textuelle multilingue. À partir d'une prémisse (`premise`) et d'une hypothèse (`hypothesis`), le modèle doit prédire la relation logique sous-jacente parmi trois classes possibles :
* **0 (Entailment / Implication)** : L'hypothèse découle nécessairement de la prémisse.
* **1 (Neutral / Neutre)** : L'hypothèse peut ou ne peut pas être vraie au vu de la prémisse.
* **2 (Contradiction)** : L'hypothèse contredit directement la prémisse.

Voici un aperçu de 3 exemples extraits du dataset illustrant la nature multilingue et les annotations :

| Prémisse | Hypothèse | Langue | Label (Classe) |
| :--- | :--- | :---: | :---: |
| *and these comments were considered in formulating the interim rules.* | *The rules developed in the interim were put together with these comments in mind.* | English | **0** (Implication) |
| *These are issues that we wrestle with in practice groups of law firms, she said.* | *Practice groups are not permitted to work on these issues.* | English | **2** (Contradiction) |
| *Des petites choses comme celles-là font une différence énorme dans ce que j'essaye de faire.* | *J'essayais d'accomplir quelque chose.* | French | **0** (Implication) |

---

### Statistiques Globales
* **Nombre total d'exemples :** 12 120 lignes
* **Nombre de classes :** 3 (0, 1, 2)
* **Nombre de langues représentées :** 15 langues (Dataset fortement multilingue avec une dominance de l'anglais, suivi du chinois, de l'arabe, du français, du swahili, de l'urdu, etc.)

### Visualisation de l'Analyse Exploratoire
Le script d'inspection a généré les distributions suivantes pour valider la structure de nos données d'entraînement :

![Analyse Exploratoire des Données (EDA)](images/EDA.png)

* Le dataset est parfaitement équilibré.
* La distribution montre que la très grande majorité des prémisses et des hypothèses ont des longueurs concentrées entre 5 et 40 mots. 
* Les hypothèses (courbe orange) sont globalement plus courtes et plus concises que les prémisses (courbe bleue).
* *Choix du `max_length` :* En combinant les longueurs de la prémisse et de l'hypothèse (qui seront concaténées sous la forme `[CLS] Prémisse [SEP] Hypothèse [SEP]`), la longueur totale cumulée dépasse rarement 80 à 100 mots. Nous optons donc pour un hyperparamètre **`max_length = 128`** tokens. Ce choix permet d'éviter toute troncature d'information tout en minimisant le coût de calcul et l'empreinte mémoire sur le GPU.

---

## 2. Choix Techniques du Modèle

Compte tenu de la nature multilingue du jeu de données (15 langues distinctes), l'utilisation d'un modèle purement anglophone comme `bert-base-uncased` ou purement francophone comme `camembert-base` est fortuit. 

Nous sélectionnons le modèle **`bert-base-multilingual-cased` (mBERT)** pré-entraîné sur 104 langues, capable de capturer efficacement les structures syntaxiques et sémantiques des différentes langues de notre dataset.