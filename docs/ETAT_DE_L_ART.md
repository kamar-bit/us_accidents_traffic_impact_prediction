# État de l'art

## Références consignées dans la documentation locale

Les notes ci-dessous sont reprises de la documentation locale, qui indiquait
avoir consulté les résumés arXiv. Cette consolidation ne constitue pas une
nouvelle lecture des articles complets ni une actualisation bibliographique.

1. **Moosavi, S., Samavatian, M. H., Parthasarathy, S., Ramnath, R. (2019)**, *A Countrywide Traffic
   Accident Dataset* : arXiv:[1906.05409](https://arxiv.org/abs/1906.05409). Présente la base
   « US-Accidents » : données spatio-temporelles et contextuelles (lieu, temps, météo, points d'intérêt,
   description en langage naturel, période jour/nuit). Le résumé que j'ai pu lire annonce environ
   2,25 millions d'accidents sur trois ans pour cette version ; le CSV utilisé dans ce projet est plus
   grande (voir [CADRAGE.md](CADRAGE.md)). Je n'ai pas lu l'article complet.
2. **Moosavi, S., Samavatian, M. H., Parthasarathy, S., Teodorescu, R., Ramnath, R. (2019)**, *Accident
   Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights* :
   arXiv:[1909.09638](https://arxiv.org/abs/1909.09638). Propose **DAP** (Deep Accident Prediction) :
   réseau récurrent pour les données dépendant du temps, couches denses pour les autres, plongements
   entraînés pour les variations spatiales. Annonce des améliorations « significatives » pour
   prédire les accidents rares. **Attention : cette tâche est la prédiction du *risque d'accident*,
   pas la prédiction de la *gravité* d'un accident donné** ; ce n'est donc pas notre tâche et leurs
   scores ne sont pas comparables aux nôtres. Je n'ai pas lu leurs chiffres.

## Points documentaires et limites

- **Les carnets Kaggle** sur la gravité : il en existe sûrement de nombreux, mais je n'en ai
  consulté aucun et ne cite donc aucun score. Avant l'oral, ouvrez trois carnets très votés et notez,
  pour chacun : la cible, le découpage (aléatoire ou temporel), les variables retenues
  (`Distance(mi)` ? `Source` ?), la métrique.
- **La licence** du jeu de données reste à vérifier sur sa fiche avant diffusion d'extraits.
- **Documentation officielle désormais vérifiée** : [fiche Kaggle du créateur](https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata). Severity va de 1 à 4 et mesure l'impact sur le trafic. La cible confirmée est `Severity >= 3`, donc 3-4 contre 1-2. Voir [DOCUMENTATION_CIBLE.md](DOCUMENTATION_CIBLE.md) pour les réserves et les proportions mesurées.

## Hypothèse à tester, pas un fait

Je soupçonne que beaucoup de résultats publics très élevés sur ce jeu de données viennent d'un
découpage aléatoire et/ou de variables mesurées après l'accident. **C'est une hypothèse de ma part,
non démontrée.** L'étude actuelle examine les variables exclues dans un audit diagnostique séparé. Elle ne démontre pas que les résultats des autres travaux proviennent de fuites.

## Comment justifier une solution originale

Vous ne battrez pas forcément un score publié : l'originalité demandée est dans la méthode.

1. **Découpage dans le temps**, avec preuve de la dérive (`results/audit/monthly_source_severity.csv`).
2. **Audit de disponibilité** : contrôle des dates météo et diagnostics distance/durée. Source2 est fixée, donc Source n'est pas un prédicteur.
3. **Ablation par groupes** : qu'est-ce qui renseigne vraiment sur l'impact (temps, lieu, météo…) ?
4. **Évaluation adaptée au déséquilibre** et intervalles de confiance.
5. Extension possible si le temps le permet : calibration des probabilités ; regroupement des zones
   par grille géographique ; variables de densité d'accidents passés de la zone (calculées sur l'apprentissage uniquement).

## Contribution retenue pour le projet

Le modèle complet à 33 variables est central. Les règles simples sont des références et les ablations expliquent ses performances. La contribution défendable est l'évaluation temporelle avec contrôle des informations admissibles, pas l'invention d'une méthode ni la recherche du minimum de variables.
