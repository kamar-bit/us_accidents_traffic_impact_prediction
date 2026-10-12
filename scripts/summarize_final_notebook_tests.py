"""Create a technical appendix and figures without overwriting the authored narrative."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/final_notebook_comparison"
WORK = ROOT / "data/final_notebook_experiments"


def count_text(value):
    return f"{int(value):,}".replace(",", " ")


def markdown(table):
    def formatted(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v).replace("|", "/")
    lines = ["| " + " | ".join(map(str, table.columns)) + " |",
             "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(formatted(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    summary = json.loads((RESULT / "run_summary.json").read_text())
    assert summary["completed_fits"] == 21
    selected = json.loads((RESULT / "frozen_selection.json").read_text())["selected"]
    validation = pd.read_csv(RESULT / "validation.csv")
    evaluation = pd.read_csv(RESULT / "historical_evaluation.csv")
    weather = pd.read_csv(RESULT / "weather.csv")
    samples = pd.read_csv(RESULT / "samples.csv")
    intervals = pd.read_csv(RESULT / "paired_validation_intervals.csv")
    cleaning = pd.read_csv(RESULT / "cleaning.csv")
    audit = json.loads((WORK / "results/audit/audit.json").read_text())
    best = max(selected, key=lambda r: r["ap"])
    frames = {s: pd.read_pickle(WORK / "data/cache" / f"{s}.pkl")
              for s in ["train", "validation", "test", "stress"]}
    frames["train"] = frames["train"].nsmallest(500000, "_priority").sort_values("_t").reset_index(drop=True)
    severity_rows, level_rows = [], []
    for split, frame in frames.items():
        valid = pd.to_datetime(frame.Start_Time, format="mixed", errors="coerce").notna() & frame.Severity.isin([1, 2, 3, 4])
        frames[split] = frame.loc[valid].reset_index(drop=True)
        frame = frames[split]
        for level, count in frame.Severity.value_counts().reindex([1, 2, 3, 4], fill_value=0).items():
            severity_rows.append({"split": split, "severity": level, "events": int(count), "fraction": count/len(frame)})
    severity = pd.DataFrame(severity_rows)
    severity.to_csv(RESULT / "severity_distribution.csv", index=False)
    for chosen in selected:
        for split in ["validation", "test", "stress"]:
            frame = frames[split]
            p = np.load(WORK / f"{chosen['name']}_{split}.npy")
            assert len(p) == len(frame) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            pred = p >= chosen["threshold"]
            budget = np.zeros(len(frame), dtype=bool)
            budget[np.lexsort((frame._priority.to_numpy(), -p))[:int(np.ceil(.1*len(frame)))]] = True
            for level in [1, 2, 3, 4]:
                mask = frame.Severity.eq(level).to_numpy()
                count = int(mask.sum())
                level_rows.append({"name": chosen["name"], "family": chosen["family"], "split": split,
                    "severity": level, "events": count,
                    "predicted_positive_fraction": float(pred[mask].mean()) if count else np.nan,
                    "selected_top10_fraction": float(budget[mask].mean()) if count else np.nan})
    levels = pd.DataFrame(level_rows)
    levels.to_csv(RESULT / "severity_level_diagnostics.csv", index=False)
    core = validation[validation.stage.eq("core")]
    main_core = core[core.after.eq(60)]
    hgb = main_core[main_core.family.eq("hgb")].sort_values("train_n")
    low, high = hgb.iloc[0], hgb.iloc[1]
    matched = validation[validation.family.eq("hgb") & validation.after.eq(60) &
                         validation.cap.eq(best["cap"]) & validation.params.eq(best["params"])].sort_values("train_n")
    matched_low, matched_high = matched.iloc[0], matched.iloc[-1]
    weather_val = weather[weather.split.eq("validation")].sort_values("after_minutes")
    strict_fraction, expanded_fraction = weather_val.eligible_fraction.tolist()
    lines = ["# Annexe technique des comparaisons du notebook final", "",
        "Série exécutée le 10 octobre 2026. Les nombres viennent des tables exportées ; aucune cellule des notebooks existants n'a été modifiée par les scripts.", "",
        "## Recommandation fondée sur la validation", "",
        f"Le meilleur candidat testé dans le scénario météo principal est **{best['family']}**, avec **{count_text(best['train_n'])} événements**, paramètres `{best['params']}`, et un plafond de **{best['cap']} catégories Weather_Condition**. AP sur validation : **{best['ap']:.4f}**. Ce résultat concerne une recherche limitée à vingt et une configurations, pas un optimum global.", "",
        f"À réglages de référence identiques (200 itérations, 31 feuilles, plafond 15), passer de 200 000 à 500 000 pour le boosting fait varier l'AP de **{low.ap:.4f} à {high.ap:.4f}**, soit **{100*(high.ap-low.ap):+.2f} points**. L'apprentissage passe de **{low.fit_seconds:.1f} à {high.fit_seconds:.1f} secondes** sur cette machine.", "",
        f"Avec la capacité d'arbres et le plafond du candidat finalement retenu, la comparaison de volume donne **{matched_low.ap:.4f} à 200 000**, contre **{matched_high.ap:.4f} à 500 000**, soit **{100*(matched_high.ap-matched_low.ap):+.2f} points**. Les mêmes paramètres et règles de préparation sont appliqués aux deux volumes.", "",
        "Le choix final doit aussi tenir compte du coût et de la simplicité : les meilleurs modèles par famille sont conservés ci-dessous pour comparer leurs compromis. Les recommandations expérimentales ne remplacent pas une décision du groupe.", "",
        "## Protocole et préparation", "",
        "Source2 uniquement ; cible positive Severity >= 3. Apprentissage 2016-2019, validation 2020, évaluation historique 2021, stabilité 2022 jusqu'au 8 septembre. Sélection par AP sur 2020, séparément par famille dans la fenêtre principale, puis paramètres et seuils figés avant 2021/2022. Les périodes historiques ont déjà été consultées dans le projet.", "",
        "Deux ensembles imbriqués de 200 000 et 500 000 événements distincts sont tirés par priorité déterministe, graine 42, sans rééquilibrage des classes. Les mêmes événements de validation servent à tous les essais. Le petit ensemble contient exactement 200 000 événements, alors que la référence historique avait 199 999 après exclusion d'un conflit : les anciens scores ne sont donc pas une reproduction exacte de cette série.", "",
        f"L'audit retrouve {count_text(audit['source2_duplicate_event_rows'])} répétitions d'événements Source2 et {audit['source2_conflicting_event_keys']} clés avec labels contradictoires. Une seule occurrence est retenue pour les répétitions ; toutes les occurrences d'une clé contradictoire sont exclues. Une réserve initiale de dix événements permet d'obtenir exactement 500 000 après filtrage, puis le sous-ensemble de 200 000.", "",
        markdown(samples), "",
        "Répartition des niveaux, en fraction des événements :", "",
        markdown(severity.pivot(index="split", columns="severity", values="fraction").reindex(["train", "validation", "test", "stress"]).reset_index()), "",
        "Le niveau 4 représente seulement 0,21 % du train, contre 34,18 % pour le niveau 3. Le niveau 1 passe de 0,04 % du train à 15,39 % en 2022. Ces différences montrent une évolution de la composition des labels ; elles ne suffisent pas à identifier une cause ou une modification officielle du fournisseur.", "",
        "Trente-quatre variables logiques : temps, coordonnées/État/fuseau, treize indicateurs routiers dont Turning_Loop, quatre indicateurs de lumière et neuf champs météo. Turning_Loop est conservée conformément au choix du groupe ; sa constance actuelle empêche un arbre d'apprendre un découpage utile sur cette variable. ID, cible et champs de fin/distance/description sont exclus des prédicteurs.", "",
        "Les bornes physiques du notebook principal sont appliquées ; les valeurs impossibles deviennent manquantes. Aucune ligne n'est supprimée en raison de météo absente. Le vocabulaire des catégories est appris uniquement sur le train de chaque configuration : fréquence minimale 100, plafond Weather_Condition 10/15/25 selon l'essai, Missing distinct de Other. Cela signifie que l'effet du volume inclut les éventuelles différences de vocabulaire, avec des règles identiques.", "",
        f"Valeurs numériques invalides corrigées ou lignes invalides exclues, total des contrôles : {int(cleaning.invalid.sum())}. Voir `cleaning.csv` pour le détail par colonne et période.", "",
        "Boosting : catégories natives, manquants numériques gérés par le modèle, taux d'apprentissage 0,1, graine 42, pas d'arrêt précoce aléatoire. Forêt : médiane pour les numériques, one-hot pour les catégories converti en matrice dense, 100 arbres, graine 42, quatre threads, aucun poids de classe. Numériques en float32 pour limiter la mémoire. Les variables binaires restent en 0/1.", "",
        "La première série utilisait une matrice creuse pour la forêt. Un pilote sur les mêmes 20 000 événements, 20 arbres et un thread a mesuré 4,65 secondes en creux contre 1,12 en dense, avec une différence maximale de prédiction nulle. Après cinq candidats terminés, la série a été interrompue pendant la forêt à 500 000, puis entièrement reprise avec des forêts en dense. Les résultats préliminaires sont séparés dans `preliminary_sparse_validation.csv`. Le tableau principal ci-dessous utilise uniquement la reprise finale, avec la même représentation pour toutes les forêts.", "",
        "AP est la métrique principale ; ROC-AUC, Brier, précision/rappel/F1, matrice de confusion et top 10 % sont complémentaires. Le seuil maximisant F1 est appris sur validation puis réutilisé ; le F1 de validation est donc optimisé sur les événements où il est affiché. Brier est descriptif et aucune calibration n'est validée. Les ex aequo du budget top 10 % sont départagés avec une priorité indépendante des labels.", "",
        "## Huit comparaisons à réglages fixes", "",
        markdown(core[["name", "family", "train_n", "after", "ap", "auc", "f1", "precision10", "recall10", "fit_seconds", "peak_process_mb"]]), "",
        "`after=0` : observation entre -120 et 0 minutes. `after=60` : observation entre -120 et +60 minutes. Les deux bornes sont incluses ; les neuf champs du relevé associé sont masqués ensemble hors fenêtre, puis leurs manquants propres sont conservés.", "",
        "![Volume et coût à réglages fixes](../results/final_notebook_comparison/volume_et_cout.png)", "",
        "## Réglages complémentaires", "",
        markdown(validation[validation.stage.eq("tuning")][["name", "family", "cap", "params", "ap", "auc", "f1", "fit_seconds"]]), "",
        "Les plafonds 10 et 25 sont comparés au plafond 15 avec le boosting de référence. Ils ne sont pas croisés avec tous les réglages d'arbres. Les variantes 15/31/63 feuilles et 200/400 itérations forment une recherche restreinte ; la forêt compare seulement des minima 5/10/20 avec 100 arbres. Aucune recherche exhaustive n'est revendiquée.", "",
        "## Vérification complémentaire du volume avec le meilleur réglage d'arbres", "",
        "Le gain associé au réglage des arbres étant plus grand que le premier gain de volume, cinq candidats supplémentaires ont été prévus : les deux volumes et les trois plafonds météo, à capacité d'arbres sélectionnée sur 2020. Le candidat 500 000/plafond 15 déjà calculé est réutilisé. Cette étape résout la comparaison de volume après réglage sans utiliser les scores de 2021 ou 2022 pour choisir. Protocole enregistré dans `followup_protocol.json` avant les cinq essais.", "",
        markdown(validation[validation.stage.eq("followup")][["name", "train_n", "cap", "params", "ap", "auc", "f1", "fit_seconds"]]), "",
        markdown(pd.read_csv(RESULT / "refined_volume_intervals.csv")[["reference", "competitor", "delta_ap", "ap_lower", "ap_upper"]]), "",
        "![Volume avec le réglage d'arbres sélectionné](../results/final_notebook_comparison/volume_apres_reglage.png)", "",
        "## Effet de la fenêtre météo", "",
        f"Sur les 50 000 événements de validation, la part de timestamps météo admissibles passe de **{100*strict_fraction:.2f} %** à **{100*expanded_fraction:.2f} %**, soit **{100*(expanded_fraction-strict_fraction):.2f} points**. Ce taux mesure l'admissibilité du timestamp, pas la complétude de chacun des neuf champs.", "",
        markdown(weather), "",
        "Le CSV contient un relevé météo associé par accident. Nous vérifions son timestamp ; nous ne recherchons ni une première mesure ni le relevé le plus récent dans un historique météo absent du fichier. Une observation après le début implique une actualisation différée. L'heure effective de publication n'est pas fournie : un relevé dans la fenêtre ne prouve pas qu'il aurait été disponible dans un système réel.", "",
        "## Différences et incertitude descriptive", "",
        markdown(intervals[["reference", "competitor", "delta_ap", "ap_lower", "ap_upper", "delta_recall10", "recall10_lower", "recall10_upper"]]), "",
        "Intervalles percentiles à 95 % issus de 200 rééchantillonnages des semaines de 2020, appariés entre prédictions. La différence est référence moins concurrent. Ils décrivent la variabilité temporelle de cette validation ; ils ne corrigent pas la sélection parmi vingt et un candidats, ni toutes les dépendances spatiales. Ne pas les présenter comme une confirmation indépendante du gagnant.", "",
        "## Modèles sélectionnés et années suivantes", "",
        markdown(pd.DataFrame(selected)[["name", "family", "train_n", "cap", "params", "ap", "auc", "f1", "threshold", "fit_seconds"]]), "",
        markdown(evaluation[["name", "split", "n", "ap", "auc", "precision", "recall", "f1", "precision10", "recall10", "brier", "threshold"]]), "",
        "![Courbes précision-rappel historiques 2021](../results/final_notebook_comparison/courbes_precision_rappel_2021.png)", "",
        "La référence constante attribue la prévalence du train à tous les événements. Son AP correspond à la prévalence de chaque période ; son ROC-AUC vaut 0,5. Son seuil F1 donne ici une prédiction positive pour tous les événements. Les scores d'années suivantes ne servent pas à changer les paramètres ou le seuil.", "",
        "Les prévalences de la cible diffèrent selon les années. Une variation d'AP ne décrit donc pas uniquement une variation de qualité du modèle : lire aussi la référence constante, ROC-AUC et les courbes précision-rappel.", "",
        "## Diagnostic des niveaux positifs pour le meilleur candidat", "",
        markdown(levels[levels.name.eq(best["name"]) & levels.severity.isin([3, 4])][["split", "severity", "events", "predicted_positive_fraction", "selected_top10_fraction"]]), "",
        "Pour les niveaux 3 et 4, la fraction déclarée positive est le rappel de ce niveau au seuil figé ; la dernière colonne est son rappel dans le budget top 10 %. Les résultats binaires agrégés sont dominés par le niveau 3 : ne pas les présenter comme une validation spécifique de la détection du niveau 4. Ce diagnostic est descriptif et ne change pas le modèle choisi. Les taux de faux positifs par niveaux 1 et 2 sont dans le CSV complet.", "",
        "## Contrôle de proximité à fiabiliser dans le futur notebook", "",
        f"L'audit relancé retrouve {count_text(audit['source2_near_collision_rows'])} collisions de proximité, alors que l'ancienne table en indiquait 533 893. Ce compteur ne sert ni au retrait des lignes ni à l'apprentissage. La formule historique convertit le timestamp en entier puis suppose des nanosecondes ; elle dépend donc de l'unité effective de la série datetime.", "",
        "Un contrôle sur les mêmes 50 000 événements du nouveau train produit 31 collisions en datetime64[ns] et 3 283 en datetime64[us], sans changer les instants. Ce test confirme la sensibilité à l'unité ; l'écart avec l'ancienne exécution est compatible avec cette cause, sans reconstruire tout son environnement. Le futur notebook devra normaliser explicitement en nanosecondes avant la division en tranches de quinze minutes. Ne pas comparer les anciens compteurs de proximité comme si leurs unités étaient garanties. Les doublons exacts et conflits de labels sont traités avec une autre clé, inchangée par ce contrôle.", "",
        "## Coût, contrôles et reproduction", "",
        f"Durée des exécutions finales avec les caches déjà préparés, préparation des entrées, vingt et un candidats, sauvegardes et évaluations compris : **{summary['seconds']/60:.1f} minutes**. Cette durée exclut la lecture/audit initial du CSV et les essais préliminaires interrompus. Le temps d'apprentissage des tableaux exclut préparation, prédiction et compression des fichiers. Le pic mémoire est le RSS total du processus, échantillonné toutes les 0,1 seconde pendant préparation/apprentissage/prédiction ; il ne mesure pas la mémoire supplémentaire du seul modèle.", "",
        f"Empreintes des trois notebooks identiques avant/après : **{summary['notebooks_unchanged']}**. Contrôles réussis : bornes météo, timestamps manquants, absence de variables interdites, conservation de Turning_Loop, vocabulaire figé, Source2, périodes, unicité des événements, absence de chevauchement, validation identique au cache historique, tailles exactes des deux trains.", "",
        "Depuis le dossier `us_accidents_traffic_impact_prediction` :", "", "```powershell",
        "python scripts/compare_final_notebook_options.py",
        "python scripts/refine_final_notebook_boosting.py",
        "python scripts/summarize_final_notebook_tests.py", "```", "",
        "Dépendances : `pip install -r requirements-experiments.txt`, dont psutil pour mesurer la mémoire. Les caches/modèles volumineux sont dans `data/final_notebook_experiments/`, ignoré par Git. Les tables, protocole, sélection figée et métadonnées sont dans `results/final_notebook_comparison/`. Les notebooks de référence et leurs résultats historiques restent intacts.", "",
        "## Rédaction du notebook", "",
        "Cette annexe conserve les mesures détaillées pour la reproduction. Le [plan du notebook](../../docs/PLAN_NOTEBOOK_FINAL.md) et le [bilan rédigé](../../docs/RESULTATS_TESTS_NOTEBOOK_FINAL.md) présentent les contributions du groupe, les choix et leurs enseignements avec quelques résultats clés.", ""]
    text = "\n".join(lines)
    text = text.replace("../results/final_notebook_comparison/", "")
    assert chr(0x2014) not in text
    (RESULT / "technical_summary.md").write_text(text, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for family, group in main_core.groupby("family"):
        group = group.sort_values("train_n")
        axes[0].plot(group.train_n/1000, group.ap, "o-", label=family)
        axes[1].plot(group.train_n/1000, group.fit_seconds, "o-", label=family)
    axes[0].set(xlabel="Apprentissage (milliers)", ylabel="Average Precision", title="Validation 2020, météo jusqu'à +60 min")
    axes[1].set(xlabel="Apprentissage (milliers)", ylabel="Secondes", title="Coût d'apprentissage, réglages fixes")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULT / "volume_et_cout.png", dpi=140)
    plt.close(fig)
    refined = validation[validation.stage.eq("followup")].copy()
    base_name = json.loads((RESULT / "followup_protocol.json").read_text())["tree_settings_from"]
    refined = pd.concat([refined, validation[validation.name.eq(base_name)]])
    fig, ax = plt.subplots(figsize=(7, 4))
    for cap, group in refined.groupby("cap"):
        group = group.sort_values("train_n")
        ax.plot(group.train_n/1000, group.ap, "o-", label=f"Plafond météo {cap}")
    ax.set(xlabel="Apprentissage (milliers)", ylabel="Average Precision",
           title="Volume après réglage des arbres, validation 2020")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULT / "volume_apres_reglage.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = frames["test"].Severity.ge(3).to_numpy(dtype="int8")
    for chosen in selected:
        p = np.load(WORK / f"{chosen['name']}_test.npy")
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=chosen["family"])
    ax.axhline(y.mean(), color="gray", linestyle="--", label="Prévalence")
    ax.set(xlabel="Rappel", ylabel="Précision", title="Évaluation historique 2021, modèles figés")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "courbes_precision_rappel_2021.png", dpi=140)
    plt.close(fig)
    print(f"Technical appendix generated. Best tested candidate: {best['name']}, validation AP {best['ap']:.6f}")


if __name__ == "__main__":
    main()
