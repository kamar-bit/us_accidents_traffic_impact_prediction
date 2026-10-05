"""Generate an evidence-backed French report, English notebook and static figures."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys

from .paths import ROOT, audit_dir, docs_dir, notebook_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .prepare import save_json


def md_table(frame):
    columns=list(frame.columns)
    rows=["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for row in frame.itertuples(index=False,name=None):
        rows.append("| "+" | ".join(str(v).replace("|",",") for v in row)+" |")
    return "\n".join(rows)


def cell(kind,text):
    d={"cell_type":kind,"metadata":{},"source":text.splitlines(keepends=True)}
    if kind=="code":
        d.update(execution_count=None,outputs=[])
    return d


def render(out=ROOT, overwrite_notebook=False):
    out=Path(out); r=out/"results"; f=out/"figures"; f.mkdir(exist_ok=True)
    audit=json.loads((audit_dir(out)/"audit.json").read_text(encoding="utf-8"))
    frozen=json.loads((r/"frozen_selection.json").read_text(encoding="utf-8"))
    val=pd.read_csv(r/"validation_models.csv").set_index("name")
    held=pd.read_csv(r/"heldout_models.csv")
    ci=pd.read_csv(r/"paired_weekly_intervals.csv")
    singles=pd.read_csv(r/"single_variables.csv")
    champion=frozen["simple_champion"]
    names=list(dict.fromkeys(["prior","best_single_rule","best_pair_rule","logistic_full",frozen["best_nonlinear_pair"],"hgb_full"]))
    supplementary=""
    extra_conclusion=""
    if (r/"geographic_pairs_intervals.csv").exists():
        extra_selection=json.loads((r/"geographic_pairs_frozen.json").read_text(encoding="utf-8"))
        geo_name=extra_selection["selected"]
        geo=pd.read_csv(r/"geographic_pairs_heldout.csv")
        geo["features"]=2
        held=pd.concat([held,geo],ignore_index=True)
        names.insert(-1,geo_name)
        geo_ci=pd.read_csv(r/"geographic_pairs_intervals.csv")
        geo_test_ci=geo_ci[geo_ci.split=="test"].iloc[0]
        geo_stress_ci=geo_ci[geo_ci.split=="stress"].iloc[0]
        supplementary=f"""
## Analyses complémentaires : références à peu de variables

Le screening des paires par arbres de quatre feuilles peut manquer une relation spatiale complexe. La forte performance du groupe géographique sur validation a donc motivé un contrôle supplémentaire de trois paires : latitude-longitude, latitude-État et longitude-État. Ces contrôles sont ajoutés après l'expérience principale et ne remplacent pas sa sélection prédéclarée. Leurs hyperparamètres reprennent ceux du modèle complet ; leur gagnant est choisi exclusivement sur 2020 avant leurs scores de test.

La paire latitude-longitude atteint une AP de {geo[(geo.split=='test')&(geo.name==geo_name)].iloc[0].ap:.4f} sur 2021. Face à ce concurrent, le gain du modèle complet en rappel est de **{100*geo_test_ci.delta_recall10:.2f} points**, IC [{100*geo_test_ci.recall10_lower:.2f} ; {100*geo_test_ci.recall10_upper:.2f}], donc inférieur au seuil de cinq points. Le gain d'AP reste de {geo_test_ci.delta_ap:.4f}. Sur 2022, le gain en rappel est de {100*geo_stress_ci.delta_recall10:.2f} points.

Il faut ainsi distinguer trois conclusions : les règles mathématiques simples testées ne reproduisent pas le modèle complet ; un modèle non linéaire à deux coordonnées est déjà performant ; toutes les autres variables combinées améliorent encore l'AP, mais leur gain opérationnel marginal dépend du budget d'alertes retenu. Le verdict favorable à la comparaison primaire ne suffit pas à dire que 33 variables sont indispensables.
"""
        extra_conclusion=f"""A supplementary check, motivated by geography-only validation performance, fits three geographic pairs and selects the best using 2020 only. Latitude and longitude alone reach AP {geo[(geo.split=='test')&(geo.name==geo_name)].iloc[0].ap:.4f} in 2021. Full boosting adds {100*geo_test_ci.delta_recall10:.2f} recall points, below the five-point usefulness threshold, while improving AP by {geo_test_ci.delta_ap:.4f}. This is a nonlinear ML model with two inputs, rather than a simple mathematical rule. The primary result does not establish that all 33 variables are necessary.
"""
    names.insert(-1,"pair_plus_geography")
    test=held[held.split=="test"].set_index("name")
    stress=held[held.split=="stress"].set_index("name")
    main_ci=ci[(ci.split=="test")&(ci.competitor==champion)].iloc[0]
    stress_ci=ci[(ci.split=="stress")&(ci.competitor==champion)].iloc[0]

    def verdict(row):
        if row.delta_recall10>=.05 and row.recall10_lower>0 and row.delta_ap>=0:
            return "Gain utile selon le critère fixé avant les résultats."
        if row.recall10_lower>0:
            return "Gain détectable, mais le critère complet d'utilité n'est pas satisfait."
        if row.recall10_upper<.05:
            return "Le gain de cinq points visé n'est pas soutenu ; privilégier la simplicité pour ce critère."
        return "Résultat indécis : l'intervalle reste compatible avec un gain utile."

    def comparison(split):
        t=held[held.split==split].set_index("name").loc[names]
        return pd.DataFrame({"Modèle":t.index,"Variables":t.features.astype(int).to_numpy(),
            "AP":t.ap.map(lambda v:f"{v:.4f}").to_numpy(),
            "Rappel, top 10 %":t.recall10.map(lambda v:f"{100*v:.2f} %").to_numpy(),
            "Précision, top 10 %":t.precision10.map(lambda v:f"{100*v:.2f} %").to_numpy()})

    # Static scientific figures use English text for the course presentation.
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    labels={"prior":"Constant","best_single_rule":"One-variable rule","best_pair_rule":"Two-variable rule",
            "logistic_full":"Full logistic",frozen["best_nonlinear_pair"]:"Two-variable boosting",
            "pair_plus_geography":"Six contextual variables","hgb_full":"Full boosting"}
    if (r/"geographic_pairs_intervals.csv").exists():
        labels[geo_name]="Coordinates only (extra)"
    fig,axes=plt.subplots(1,2,figsize=(11,5.1))
    for ax,metric,title in zip(axes,["ap","recall10"],["Average Precision (2021)","Recall in top 10% of alerts (2021)"]):
        ax.barh([labels[n] for n in names],test.loc[names,metric],color=["#8b949e"]*(len(names)-1)+["#2474b5"])
        ax.invert_yaxis(); ax.set_xlim(0, max(test.loc[names,metric])*1.24); ax.set_title(title)
        for i,n in enumerate(names):
            value=test.loc[n,metric]; ax.text(value+.004,i,f"{value:.3f}",va="center")
    fig.suptitle("Source2: untouched 2021 evaluation")
    fig.tight_layout(); fig.savefig(f/"01_rules_vs_ml.png",dpi=150); plt.close(fig)

    budgets=[n for n in val.index if n.startswith("budget_")]
    budgets=sorted(budgets,key=lambda n:int(n.split("_")[-1]))
    fig,ax=plt.subplots(figsize=(7,4))
    for table,label in [(val,"Validation 2020"),(test,"Test 2021"),(stress,"Stress 2022")]:
        ax.plot([int(n.split("_")[-1]) for n in budgets],table.loc[budgets,"ap"],"o-",label=label)
    ax.set(xlabel="Number of logical input variables",ylabel="Average Precision",title="Does more information help?")
    ax.legend(); fig.tight_layout(); fig.savefig(f/"02_feature_budget.png",dpi=150); plt.close(fig)

    group_rows=[]
    for group in ["time","geography","road","light","weather"]:
        group_rows.append({"group":group,"validation_delta_ap":val.loc["hgb_full","ap"]-val.loc[f"without_{group}","ap"],
                           "test_delta_ap":test.loc["hgb_full","ap"]-test.loc[f"without_{group}","ap"]})
    gr=pd.DataFrame(group_rows)
    gr.to_csv(r/"group_ap_gains.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    positions=np.arange(len(gr))
    ax.barh(positions-.18,gr.validation_delta_ap,height=.35,label="Validation 2020")
    ax.barh(positions+.18,gr.test_delta_ap,height=.35,label="Test 2021")
    ax.set_yticks(positions,gr.group); ax.axvline(0,color="black",lw=.8)
    ax.set(xlabel="Full AP minus AP without this group",title="Incremental information from each group")
    ax.legend(); fig.tight_layout(); fig.savefig(f/"03_group_ablation.png",dpi=150); plt.close(fig)

    learning=sorted([n for n in val.index if n.startswith("train_size_")],key=lambda n:int(n.split("_")[-1]))
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([int(n.split("_")[-1]) for n in learning],val.loc[learning,"ap"],"o-")
    ax.set(xlabel="Training examples",ylabel="2020 Average Precision",title="Learning curve with nested uniform samples")
    fig.tight_layout(); fig.savefig(f/"04_learning_curve.png",dpi=150); plt.close(fig)

    monthly=pd.read_csv(r/"monthly_state_scores.csv")
    fig,ax=plt.subplots(figsize=(10,4))
    for name in ["hgb_full",champion]:
        z=monthly[(monthly.dimension=="month")&(monthly.name==name)].sort_values("value")
        ax.plot(z.value,z.ap,"o-",label=name)
    ax.tick_params(axis="x",rotation=60); ax.set(ylabel="Average Precision",title="Performance through 2021 and available 2022 months")
    ax.legend(); fig.tight_layout(); fig.savefig(f/"05_monthly_performance.png",dpi=150); plt.close(fig)

    source=pd.read_csv(audit_dir(out)/"monthly_source_severity.csv")
    source["positives"]=source["count"]*(source.Severity>=3)
    drift=source.groupby(["_month","Source"])[["count","positives"]].sum().reset_index()
    fig,ax=plt.subplots(figsize=(10,4))
    for name,z in drift.groupby("Source"):
        ax.plot(pd.to_datetime(z._month),z.positives/z["count"],label=name)
    ax.set(ylabel="Proportion with Severity >= 3",title="Collection shifts justify the Source2 study scope")
    ax.legend(); fig.tight_layout(); fig.savefig(f/"06_source_drift.png",dpi=150); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4))
    z=ci[ci.split=="test"].copy()
    if (r/"geographic_pairs_intervals.csv").exists():
        z=pd.concat([z,geo_ci[geo_ci.split=="test"]],ignore_index=True)
    if (r/"sensitivity_intervals.csv").exists():
        extra_intervals=pd.read_csv(r/"sensitivity_intervals.csv")
        z=pd.concat([z,extra_intervals[(extra_intervals.split=="test")&(extra_intervals.competitor=="budget_5")]],ignore_index=True)
    positions=np.arange(len(z))
    ax.errorbar(z.delta_recall10,positions,xerr=np.vstack([z.delta_recall10-z.recall10_lower,z.recall10_upper-z.delta_recall10]),fmt="o")
    tick_labels=[labels.get(n,"Five-variable boosting" if n=="budget_5" else n) for n in z.competitor]
    ax.set_yticks(positions,tick_labels); ax.axvline(0,color="gray",ls=":"); ax.axvline(.05,color="#c06b20",ls="--")
    ax.set(xlabel="Full boosting minus competitor: recall in top 10%",title="Paired weekly 95% intervals on 2021")
    fig.tight_layout(); fig.savefig(f/"07_paired_intervals.png",dpi=150); plt.close(fig)

    single_rule=frozen["models"]["best_single_rule"]["rule"]
    best_pair=frozen["models"]["best_pair_rule"]
    (r/"best_pair_rule.txt").write_text(best_pair["tree"],encoding="utf-8")
    category_rates=pd.DataFrame(single_rule.get("rates",{}).items(),columns=["category","probability"]).sort_values("probability",ascending=False)
    category_rates.to_csv(r/"best_single_category_rates.csv",index=False)
    pair_count=len(pd.read_csv(r/"pair_rules.csv"))
    verification=json.loads((r/"verification.json").read_text(encoding="utf-8"))
    diag=held[(held.split=="test")&held.name.str.startswith("diagnostic_")]
    near=pd.read_csv(r/"near_duplicate_sensitivity.csv")
    top=singles.head(10)[["variable","group","ap"]].copy(); top.ap=top.ap.map(lambda v:f"{v:.4f}")
    change=100*main_ci.delta_recall10
    p_ap=main_ci.delta_ap
    full_test=test.loc['hgb_full']
    full_stress=stress.loc['hgb_full']
    headline=f"Le modèle complet à 33 variables atteint sur 2021 une AP de {full_test.ap:.4f}, une ROC-AUC de {full_test.auc:.4f} et une précision de {100*full_test.precision10:.2f} % parmi les 10 % d'incidents les mieux classés. Sur la période disponible de 2022, son AP reste de {full_stress.ap:.4f}."
    target_documentation=(docs_dir(out)/"DOCUMENTATION_CIBLE.md").read_text(encoding="utf-8").split("\n",1)[1].strip()
    central=test.loc[['prior','logistic_full','hgb_full'],['ap','auc','recall10','precision10']].reset_index().round(4)
    grouptext="\n".join(f"- {row.group} : gain d'AP de {row.test_delta_ap:.4f} sur 2021, en comparant le modèle complet à sa variante sans ce groupe." for row in gr.itertuples())
    compact=test.loc["pair_plus_geography"]
    shuffled=test.loc["shuffled_labels_control"]
    extra_null=""
    if (r/"shuffled_label_replicates.csv").exists():
        null=pd.read_csv(r/"shuffled_label_replicates.csv")
        null_test=null[null.split=="test"]
        extra_null=f"Cinq graines de mélange donnent sur 2021 une AP comprise entre {null_test.ap.min():.4f} et {null_test.ap.max():.4f}, et une AUC entre {null_test.auc.min():.4f} et {null_test.auc.max():.4f}. Ces faibles écarts au hasard ne sont pas exactement nuls, mais restent loin de l'AP du modèle réel. Voir `shuffled_label_replicates.csv`."
    extra_recall=""
    if (r/"sensitivity_intervals.csv").exists():
        sensitivity=pd.read_csv(r/"sensitivity_intervals.csv")
        alt=json.loads((r/"additional_checks.json").read_text(encoding="utf-8"))["recall_selected_simple_competitor"]
        rc=sensitivity[(sensitivity.split=="test")&(sensitivity.competitor==alt)].iloc[0]
        extra_recall=f"\nUne autre sensibilité choisit le concurrent à deux variables sur le rappel de 2020 au lieu de l'AP : {alt}. Le gain du modèle complet sur 2021 est alors de {100*rc.delta_recall10:.2f} points, IC [{100*rc.recall10_lower:.2f} ; {100*rc.recall10_upper:.2f}]. Le choix de la métrique de sélection modifie donc l'interprétation opérationnelle.\n"
    report=f"""# Prédire l'impact sur le trafic au début d'un accident

{headline}

## Résultat principal : modèle complet

Question : peut-on identifier au début d'un accident les incidents classés aux niveaux 3 ou 4 d'impact sur le trafic, à partir du contexte, et conserver les performances l'année suivante ?

Le modèle principal est le boosting complet à 33 variables admissibles. Les règles simples sont des références de non-trivialité ; les ablations expliquent le modèle. La réduction du nombre de variables n'est pas l'objectif central. Ce recadrage intervient après les expériences et ne change ni les scores ni les choix figés.

{md_table(central)}

AP et AUC sont des mesures de classement, pas des pourcentages de prédictions correctes. Dans les 10 % de scores les plus élevés sur 2021, {100*full_test.precision10:.2f} % des événements sont positifs ; ce budget retrouve {100*full_test.recall10:.2f} % de tous les positifs. Le modèle n'identifie donc pas tous les incidents d'impact supérieur. Les probabilités n'ont pas été validées par calibration.

## Documentation de la cible confirmée

Voir [DOCUMENTATION_CIBLE.md](DOCUMENTATION_CIBLE.md). Son contenu est repris ici pour que le rapport reste autonome.

{target_documentation}

## Protocole et données

Les comparaisons temporelles évaluent le modèle complet, pas une recherche de la plus petite liste de variables.

L'audit a lu {audit['rows']:,} lignes et {audit['columns']} colonnes. La tâche porte sur l'impact trafic conditionnellement à un accident enregistré, avec la cible binaire `Severity >= 3`. Elle ne mesure pas la gravité corporelle.

Source2 uniquement. Train 2016-2019 ({audit['samples']['train']['rows']:,} événements), validation 2020 ({audit['samples']['validation']['rows']:,}), test 2021 ({audit['samples']['test']['rows']:,}), stress 2022 ({audit['samples']['stress']['rows']:,}). Source2 s'arrête au {audit['samples']['stress']['last'][:10]} dans ce fichier : le stress ne couvre pas toute l'année 2022.

Échantillonnage uniforme d'événements distincts par priorités de hachage, graine 42, sans équilibrage des classes. {audit['source2_duplicate_event_rows']:,} répétitions d'événement et {audit['source2_conflicting_event_keys']} clés contradictoires identifiées sur Source2. Les répétitions sont regroupées ; les clés contradictoires sont exclues. Les tailles peuvent être légèrement inférieures aux plafonds après exclusions. Identifiants dupliqués : {audit['duplicate_ids']}.

La clé exacte inclut source, début, fin, coordonnées de départ, distance et description. Ces champs servent à reconnaître un enregistrement répété ; ils ne sont pas tous des prédicteurs. Les hachages 64 bits présentent un risque négligeable de collision. La sensibilité aux collisions dans des cellules de 0,001 degré et 15 minutes est rapportée séparément : elles ne prouvent pas l'identité de deux accidents.

Les transformations apprennent uniquement sur le train. La météo est masquée si son observation n'appartient pas à [Start_Time - 120 minutes, Start_Time]. Sur le CSV complet, {100*audit['weather']['after_start']/audit['weather']['observed']:.2f} % des observations météo datées sont postérieures au début. Aucune ligne n'est retirée pour ce motif. L'horaire de publication des observations reste inconnu : il s'agit d'une hypothèse de disponibilité, pas d'un simulateur de premier signalement.

`Country` et `Turning_Loop` sont constants dans le fichier. Variables constantes supplémentaires supprimées sur le train : {', '.join(frozen['dropped_train_constants'])}. Nombre de variables admissibles : {len(frozen['feature_order'])}. Aucun texte libre, adresse fine, identifiant, source, durée ou distance dans les concurrents principaux.

## Diagnostics complémentaires : règles et relations mathématiques

{len(singles)} règles univariées et {pair_count} paires ont été évaluées sur la même validation. Chaque règle numérique cherche le seuil maximisant l'AP sur le train, avec au moins 1 % des lignes de chaque côté. Les tables catégorielles sont lissées avec 50 observations de poids équivalent vers la moyenne du train. Les valeurs inconnues utilisent cette moyenne. Les arbres de paire ont au plus quatre feuilles et trois niveaux, avec au moins 1 % du train dans chaque feuille.

Meilleure variable pour la règle univariée : **{frozen['best_single']}**. Meilleure paire pour l'arbre simple : **{', '.join(frozen['best_pair'])}**. Une variable catégorielle peut représenter plusieurs modalités : le nombre de variables ne résume pas toute la complexité de la règle. Les taux par catégorie sont dans `results/best_single_category_rates.csv`.

Arbre de la meilleure paire (les modalités sont encodées en indicatrices, les poids indiquent les classes 0 et 1) :

```text
{best_pair['tree'].rstrip()}
```

Les régressions logistiques à une variable, deux variables et toutes les variables testent des relations additives. Le boosting sur les cinq meilleures paires teste des relations plus complexes avec très peu d'entrées. Cela évite de conclure que beaucoup de variables sont nécessaires simplement parce qu'un seuil isolé est trop restrictif.

Meilleures variables isolées sur 2020 :

{md_table(top)}

## Test final 2021

{md_table(comparison('test'))}

Référence AP sans information : prévalence de {test.loc['prior','prevalence']:.4f}. Le rappel à 10 % sélectionne un budget identique pour tous les modèles, avec égalités départagées par une priorité indépendante de la cible. C'est un classement rétrospectif. La probabilité constante conserve le taux du train ; sa calibration peut différer de celle du test.

Face à {champion}, différence de rappel : {change:.2f} points, IC 95 % [{100*main_ci.recall10_lower:.2f} ; {100*main_ci.recall10_upper:.2f}]. Différence d'AP : {p_ap:.4f}, IC [{main_ci.ap_lower:.4f} ; {main_ci.ap_upper:.4f}]. Intervalles appariés par rééchantillonnage de semaines, {int(main_ci.draws)} tirages.

Le critère historique d'utilité comparant la complexité avait été fixé à cinq points supplémentaires de rappel, un intervalle de différence excluant zéro et aucune baisse d'AP. Il est conservé pour interpréter cette expérience secondaire, pas comme nouveau critère de réussite de la question prédictive. Il est pédagogique, sans interprétation économique. Un intervalle qui englobe zéro ne démontre pas une équivalence.

{supplementary}
{extra_recall}

## Apport des groupes et du volume

{grouptext}

Le modèle à six variables (`hour`, `Traffic_Signal`, `Start_Lat`, `Start_Lng`, `State`, `Timezone`) atteint une AP de {compact.ap:.4f} et un rappel de {100*compact.recall10:.2f} % sur 2021. Le modèle complet atteint {test.loc['hgb_full','ap']:.4f} et {100*test.loc['hgb_full','recall10']:.2f} %. Ce diagnostic montre des redondances ; il ne démontre pas un optimum et ne remplace pas le modèle complet comme résultat principal.

Ces différences sont des associations prédictives conditionnelles aux autres entrées, pas des effets causaux de la météo ou de la route. Les variantes utilisent les hyperparamètres du modèle complet, sans réglage spécifique à chaque ablation. Les permutations de groupes peuvent créer des combinaisons irréalistes et doivent être lues avec les ablations réentraînées. L'ordre de la courbe de variables a été fixé sur validation, en partant de la meilleure paire puis des règles univariées les mieux classées. Ce n'est pas une recherche exhaustive de tous les sous-ensembles de cinq ou dix variables.

Les figures `02_feature_budget.png`, `03_group_ablation.png` et `04_learning_curve.png` montrent le gain par budget, par groupe et par taille du train. Le modèle d'indicateurs de manquants est dans la table complète : il peut signaler des particularités de collecte.

## Stress 2022 et contrôles

{md_table(comparison('stress'))}

{verdict(stress_ci)} Face au même concurrent figé : {100*stress_ci.delta_recall10:.2f} points de rappel, IC [{100*stress_ci.recall10_lower:.2f} ; {100*stress_ci.recall10_upper:.2f}]. Aucun réglage sur cette période. Les résultats mensuels et par État sont dans `monthly_state_scores.csv` ; ne pas comparer des États sans considérer leur prévalence et leur effectif.

Contrôle d'étiquettes mélangées : AP {shuffled.ap:.4f}, AUC {shuffled.auc:.4f} sur 2021. L'AP attendue sans information est proche de la prévalence. Des écarts finis existent ; ce contrôle seul ne garantit pas l'absence de toute fuite.

{extra_null}

Audit des variables exclues sur 2021 :

{md_table(diag[['name','ap','recall10']].round(4))}

Ces scores diagnostiques ne sont pas en compétition avec les variables autorisées. Leur association avec la cible ne prouve pas à elle seule une fuite, et leur faible performance ne garantit pas leur disponibilité initiale. Les règles à quatre classes sur distance et durée sont décrites dans `future_information_4class.csv`.

Sensibilité aux collisions spatiotemporelles :

{md_table(near[['split','name','excluded_within_holdout','ap','recall10']].round(4))}

L'entraînement avec un représentant par cellule est comparé au modèle complet. Les collisions avec le train sont nulles ou rares par construction des périodes disjointes ; la déduplication interne des périodes est le contrôle informatif ici.

## Reproductibilité, originalité et limites

Sélection figée avant calcul des scores de 2021 et 2022 dans `frozen_selection.json`. Modèles sauvegardés dans `frozen_models.joblib`, prédictions et priorités dans `predictions_test.csv` et `predictions_stress.csv`. Hyperparamètres HGB : {json.dumps(frozen['hgb_parameters'])}. Sans poids de classe et sans arrêt anticipé utilisant un split aléatoire. Durée des expériences : {verification['elapsed_seconds']:.0f} secondes. Versions : {json.dumps(frozen['versions'])}.

La contribution du projet est une prédiction par modèle complet avec évaluation temporelle, contrôle des informations admissibles et analyse des limites. Les règles et budgets d'information sont des diagnostics de non-trivialité. Le sujet et l'évaluation temporelle existent déjà : [projet comparable](https://github.com/Simranghag22/us-accidents-severity-prediction), [autre projet de cours](https://github.com/JustinJ-Lo/vehicle-severity-ml). Leurs scores ne sont pas directement comparables aux nôtres. Définition de la cible : [fiche du créateur](https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata).

Limites : fournisseur unique, informations de trafic et type exact de route absents, disponibilité au signalement non garantie, observations limitées aux accidents collectés. Une faible performance ne prouve pas l'absence de signal ni l'inutilité universelle du ML. La fréquence du trafic et le nombre d'accidents sans exposition routière ne permettent pas d'estimer le risque de circulation.

Les cellules actualisées en anglais sont proposées dans CELLULES_NOTEBOOK.md. Le notebook existant conserve son ancien récit tant qu'elles ne sont pas intégrées manuellement ; il n'est pas remplacé par défaut. L'archive reste donc un dossier de travail, pas un rendu final recadré. Ajouter les noms des membres, joindre les modules et ne pas joindre le CSV ou les caches. Limite du notebook : 10 Mo.
"""
    assert chr(0x2014) not in report
    (docs_dir(out)/"RAPPORT.md").write_text(report,encoding="utf-8")
    save_json(r/"verdict.json",{"simple_champion":champion,"2021":verdict(main_ci),"2022":verdict(stress_ci),
        "2021_delta_recall10":float(main_ci.delta_recall10),"2021_delta_ap":float(main_ci.delta_ap)})

    setup='''from pathlib import Path
import sys
ROOT = Path.cwd()
while not (ROOT / "traffic_impact").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / "traffic_impact").exists(), "Open this notebook inside the repository."
sys.path.insert(0, str(ROOT))
from traffic_impact.experiment import FeatureBuilder, Model, score, threadpool_limits
import pandas as pd
import json
RESULTS = ROOT / "results"
'''
    audit_code='''audit = json.loads((ROOT / "results" / "audit" / "audit.json").read_text(encoding="utf-8"))
display(pd.DataFrame(audit["samples"]).T)
display(pd.read_csv(ROOT / "results" / "audit" / "columns.csv").sort_values("missing_pct", ascending=False).head(12))
'''
    train_code='''# The full reproducible pipeline audits, fits, freezes selection, and evaluates.
# Set True only when intentionally rerunning the complete experiment.
RUN_FULL_STUDY = False
if RUN_FULL_STUDY:
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_study.py"), "--rerun"], check=True)
'''
    feature_code='''raw_train = pd.read_pickle(ROOT / "data" / "cache" / "train.pkl")
builder = FeatureBuilder().fit(raw_train)
X_train = builder.transform(raw_train)
y_train = (raw_train["Severity"] >= 3).astype(int).to_numpy()
display(pd.Series(builder.groups, name="group").to_frame())
assert not {"Severity", "Source", "ID", "Distance(mi)", "duration_min"} & set(X_train.columns)
'''
    feature_code='''selection = json.loads((RESULTS / "frozen_selection.json").read_text(encoding="utf-8"))
if (ROOT / "data" / "cache" / "train.pkl").exists():
    raw_train = pd.read_pickle(ROOT / "data" / "cache" / "train.pkl")
    builder = FeatureBuilder().fit(raw_train)
    X_train = builder.transform(raw_train)
    y_train = (raw_train["Severity"] >= 3).astype(int).to_numpy()
    assert not {"Severity", "Source", "ID", "Distance(mi)", "duration_min"} & set(X_train.columns)
display(pd.Series(selection["feature_groups"], name="group").to_frame())
'''
    rule_code='''selection = json.loads((RESULTS / "frozen_selection.json").read_text(encoding="utf-8"))
display(pd.read_csv(RESULTS / "single_variables.csv").head(10))
display(pd.read_csv(RESULTS / "pair_rules.csv").head(5))
print((RESULTS / "best_pair_rule.txt").read_text(encoding="utf-8"))
display(pd.read_csv(RESULTS / "best_single_category_rates.csv"))
'''
    eval_code='''scores = pd.read_csv(RESULTS / "heldout_models.csv")
display(scores[scores["name"].isin(["prior", "best_single_rule", "best_pair_rule",
    "logistic_full", selection["best_nonlinear_pair"], "hgb_full"])])
display(pd.read_csv(RESULTS / "paired_weekly_intervals.csv"))
display(pd.read_csv(RESULTS / "group_ap_gains.csv"))
display(pd.read_csv(RESULTS / "near_duplicate_sensitivity.csv"))
display(pd.read_csv(RESULTS / "geographic_pairs_heldout.csv"))
display(pd.read_csv(RESULTS / "geographic_pairs_intervals.csv"))
'''
    figure_code='''from IPython.display import Image, display
for name in ["01_rules_vs_ml.png", "02_feature_budget.png", "03_group_ablation.png",
             "04_learning_curve.png", "05_monthly_performance.png", "06_source_drift.png",
             "07_paired_intervals.png"]:
    display(Image(filename=str(ROOT / "figures" / name)))
'''
    conclusion=f'''The full 33-input model predicts the recorded upper traffic-impact levels (Severity 3 or 4), rather than injury severity or delay minutes. On the 2021 temporal holdout it reaches AP {full_test.ap:.4f} and ROC-AUC {full_test.auc:.4f}, compared with AP {test.loc['logistic_full','ap']:.4f} for full logistic regression and {test.loc['prior','ap']:.4f} for the no-information reference.

Within the highest-ranked 10% of incidents, precision is {100*full_test.precision10:.2f}% and recall is {100*full_test.recall10:.2f}%. On the available 2022 period, AP remains {full_stress.ap:.4f}. This supports temporal ranking performance in the studied provider, not exhaustive detection or deployment readiness. Probabilities have not been validated by calibration.

Simple rules, feature budgets and geographic pairs are secondary diagnostics. They do not establish that all 33 inputs are necessary. The historical five-recall-point criterion judged incremental complexity; it is not a newly preregistered success criterion for this reframed predictive question. The framing changed after the experiments, with scores and frozen selection preserved.

Available Source2 observations end on {audit['samples']['stress']['last'][:10]}. Most positive incidents are Severity 3; the binary scores alone do not establish good detection of Severity 4.

This study establishes predictive comparisons for recorded incidents from one provider. Group ablations do not establish causal effects, and timestamp filtering does not guarantee that weather was published at first report. With more time we would investigate road type, independent providers, repeated time windows and probability calibration.
'''
    cells=[cell("markdown","# Predicting traffic impact at the start of an accident\n\nGroup members: [ADD ALL NAMES]\n\n## Introduction\n\nCan a full context model identify incidents assigned upper traffic-impact levels at accident start, and do its performances persist in the following year?\n\nThe full model combines 33 admissible inputs. Simple rules are benchmarks, not the main objective. Our contribution is temporal evaluation with information-availability checks and model diagnostics, not a new algorithm. This framing was revised after the experiments without changing frozen choices or scores.\n\n## Target definition\n\nSeverity is an ordered scale from 1 to 4 describing traffic impact. The creator describes 1 as the least impact with short delay and 4 as significant impact with long delay. No precise delay-minute boundaries between levels are supplied. This is not injury severity.\n\nWe retain binary y = (Severity >= 3): levels 3 and 4 are positive, levels 1 and 2 negative. High impact is our study grouping of the upper levels, not an official operational cutoff. About 33.89% of Source2 records in 2016-2021 are positive before cleaning; about 98.77% of these positives are level 3. Binary scores do not establish good level-4 detection.\n\nStart_Time is the retrospective prediction-time proxy. Publication times and the time Severity was assigned are unknown, so first-report deployment is not validated.\n\nSource: https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata (accessed 2026-10-03). Detailed source-grounded target documentation accompanies this notebook in DOCUMENTATION_CIBLE.md.\n\nRelated work: https://github.com/Simranghag22/us-accidents-severity-prediction and https://github.com/JustinJ-Lo/vehicle-severity-ml. Their scores are not directly comparable to ours.\n"),
        cell("code",setup),cell("markdown","## Preprocessing\n\nFull streaming audit; uniformly sampled distinct events from Source2. Train: 2016-2019, validation: 2020, untouched test: 2021, stress: available 2022. Weather observations outside the preceding two-hour window are masked. Future information is diagnostic only. Category vocabularies, numeric imputations, thresholds and constant removal are fitted on training data. Conflicting duplicate-event labels are excluded.\n"),
        cell("code",audit_code),cell("code",feature_code),cell("markdown","## Training\n\nThe central comparison is a constant prior, full logistic regression and full histogram gradient boosting with all 33 admissible inputs. Select hyperparameters using 2020 validation AP and freeze choices before future-year evaluation. Single-variable rules, pair trees and feature budgets are supplementary non-triviality and redundancy diagnostics.\n\nThe modules prepare.py, experiment.py, run_study.py and render_report.py and DOCUMENTATION_CIBLE.md must accompany this notebook. Run `pip install -r requirements.txt` in your environment.\n"),
        cell("code",train_code),cell("code",rule_code),cell("markdown","## Evaluation\n\nPrimary metrics are Average Precision and recall in the highest-ranked 10% of incidents, supplemented by precision and ROC-AUC. Resolve ties reproducibly without target information. Compare models on the same samples. Paired weekly bootstrap intervals use 500 draws. Group ablations explain incremental information, not causal effects. Probabilities are not validated by calibration.\n\nThe five-recall-point criterion belongs to the historical complexity comparison, a secondary analysis rather than an operational guarantee or a new success criterion for the reframed prediction task.\n"),
        cell("code",eval_code),cell("code",figure_code),cell("markdown","## Conclusion\n\n"+conclusion)]
    # Static results and attached figures make the report readable without running cells.
    english_test=test.loc[names,["features","ap","recall10","precision10"]].reset_index().rename(columns={"name":"model"}).round(4)
    cells.insert(-1,cell("markdown","### Recorded 2021 results\n\n"+md_table(english_test)+"\n\nGeographic pairs are supplementary checks; the primary selection remains frozen.\n"))
    for path in sorted(f.glob("*.png")):
        c=cell("markdown",f"![{path.stem}](attachment:{path.name})\n")
        c["attachments"]={path.name:{"image/png":base64.b64encode(path.read_bytes()).decode("ascii")}}
        cells.insert(-1,c)
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
    for i,c in enumerate(cells): c["id"]=f"study-{i:02d}"
    nb_path=notebook_path(out)
    nb_path.parent.mkdir(parents=True,exist_ok=True)
    if overwrite_notebook or not nb_path.exists():
        save_json(nb_path,notebook)
    else:
        print("Existing notebook preserved. Updated cells are in CELLULES_NOTEBOOK.md.",flush=True)
    assert nb_path.stat().st_size<10000000
    # A Markdown copy gives complete cells for copying without editing a notebook.
    copy_cells=[]
    for i,c in enumerate(cells):
        language="python" if c["cell_type"]=="code" else "markdown"
        source="".join(c['source']).rstrip()
        if c.get("attachments"):
            source=source.replace("attachment:","../figures/")
        copy_cells.append(f"## Cell {i+1} ({c['cell_type']})\n\n```{language}\n{source}\n```")
    (docs_dir(out)/"CELLULES_NOTEBOOK.md").write_text("# Complete proposed notebook cells\n\nThe existing notebook is not modified. Copy the English Markdown and Python cells below to update it. Image attachments are shown as relative figure paths in this copy.\n\n"+"\n\n".join(copy_cells),encoding="utf-8")
    print(report[:1200],flush=True)
    return docs_dir(out)/"RAPPORT.md"


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=ROOT)
    p.add_argument("--overwrite-notebook",action="store_true",help="Explicitly replace an existing notebook; otherwise preserve it.")
    a=p.parse_args(); render(a.out,overwrite_notebook=a.overwrite_notebook)
