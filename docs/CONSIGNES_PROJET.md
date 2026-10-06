# Consignes du projet : Hands-on Machine Learning

Transcription structurée du rappel fourni par le groupe dans cette conversation.
Ce fichier sert de référence de travail ; il ne remplace pas une annonce
officielle ultérieure des enseignants.

**Cours :** Master 1 Computer Science – Artificial Intelligence.
**Coordination et cours :** Marc EVRARD.
**Travaux pratiques :** Yue MA et Quentin LE TELLIER.
**Période indiquée :** septembre–octobre.
**Note du module :** contrôle continu 20 % (deux quiz) et projet 80 %.

## Présentation

- Présentation **en anglais**.
- **10 minutes d'exposé**, puis **10 minutes de questions**.
- Tous les membres du groupe doivent participer de façon équitable.
- Les slides sont recommandées plutôt que la projection du notebook brut.

## Fichiers à rendre

- **Un seul notebook par groupe**, contenant le code et le rapport en Markdown.
- Si les slides sont distinctes du notebook, **un seul PDF de slides par groupe**.
- Inclure **les noms de tous les membres** dans le notebook et les slides.
- Joindre les modules Python externes si le projet en utilise.
- **Ne pas joindre les données** ; mettre leur lien dans le notebook.
- Garder le notebook sous **10 Mo** ; éviter notamment les sorties interactives
  lourdes, comme celles que peut produire Plotly.

## Structure demandée du rapport

1. **Introduction :** expliquer la tâche avec nos propres mots.
2. **Preprocessing :** exploration, nettoyage et préparation des données.
3. **Training :** méthodes et apprentissage des modèles.
4. **Evaluation :** performances et analyse des résultats.
5. **Conclusion :** ce qui a été fait, ce qui a fonctionné ou non, et ce que
   nous ferions avec davantage de temps.

## Grille d'évaluation annoncée

| Élément | Points |
| --- | ---: |
| Preprocessing : exploration et nettoyage | /5 |
| Modèle | /5 |
| Lisibilité du code | /5 |
| Évaluation des performances, analyse et explication | /5 |
| Oral et réponses aux questions, note individuelle | /20 |

Cette grille concerne la présentation du projet, prévue sur 10 minutes
d'exposé et 10 minutes de questions. Le rappel ne précise pas ici comment ces
points sont convertis en la note globale du projet.

## Recommandations pour choisir et défendre le problème

- Privilégier un **problème tabulaire** ; éviter le NLP ou le traitement
  avancé d'images pour ce cours.
- Choisir une cible **non triviale** : elle ne doit pas être déduite par un
  simple calcul ni être presque directement contenue dans une variable.
  Préférer une cible difficile à obtenir à partir de variables plus faciles
  à mesurer.
- Viser **plus de 10 variables** (idéalement plus de 100) et **plus de 1 000
  exemples** (idéalement entre 10 000 et 1 million).
- Les variables et les classes doivent être **compréhensibles et explicites**.
- Résumer les solutions existantes et leur **état de l'art** dans le notebook
  ou les slides, par exemple à partir de Codabench ou Kaggle.
- Expliquer pourquoi notre solution a été choisie et en quoi notre travail
  apporte une contribution **originale**. Comparer honnêtement les données,
  les variables autorisées et les protocoles avant de comparer des scores.

## Application au projet US Accidents

La question, la cible et les limites sont décrites dans [CADRAGE.md](CADRAGE.md)
et [DOCUMENTATION_CIBLE.md](DOCUMENTATION_CIBLE.md). Le
[protocole](PROTOCOLE.md) et le [rapport](RAPPORT.md) documentent les expériences.
Le [mode d'emploi](MODE_EMPLOI.md) prépare la reproduction et l'oral.

Le notebook en anglais existe dans le dossier notebooks/. Ajouter les noms
de tous les membres et vérifier le rendu avant remise. Les slides restent
à préparer si le groupe choisit ce support. Ces consignes sont une transcription
fournie par le groupe, sans nouvelle vérification auprès des enseignants.
