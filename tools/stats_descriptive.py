import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def analyse_dataframe(df: pd.DataFrame):
    for col in df.columns:
        print(f"\n--- Colonne : {col} ---")
        
        # Quantitative (numérique)
        if pd.api.types.is_numeric_dtype(df[col]):
            print("Type : Quantitative")
            print(df[col].describe())
        
        # Qualitative (catégorielle ou texte)
        else:
            print("Type : Qualitative")
            print(df[col].value_counts())

def analyse_graphique(df: pd.DataFrame, target: str, seuil_modalites: int = 15, nb_bins: int = 10):
    """
    Analyse graphique des variables explicatives par rapport à la cible.
    
    - Si la cible est continue :
        * Scatterplot pour les quantitatives continues (subplot groupé)
        * Barplot (moyenne de la cible) pour les qualitatives ou quantitatives discrètes (<= seuil_modalites)
    
    - Si la cible est binaire (0/1) :
        * Barplot (taux moyen de 1) pour toutes les variables (quanti continues binnées, quali, discrètes)
    """
    cible_binaire = df[target].nunique(dropna=True) == 2

    quant_cont = []
    quant_discr_cat = []

    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= seuil_modalites:
                quant_discr_cat.append(col)  # numérique discret → comme qualitatif
            else:
                quant_cont.append(col)
        else:
            quant_discr_cat.append(col)

    # --- Variables quantitatives continues ---
    if len(quant_cont) > 0:
        print("\n--- Variables quantitatives continues ---")
        n = len(quant_cont)
        n_cols = 2
        n_rows = (n + 1) // n_cols

        plt.figure(figsize=(n_cols * 6, n_rows * 4))

        for i, col in enumerate(quant_cont):
            plt.subplot(n_rows, n_cols, i + 1)

            if cible_binaire:
                # Discrétisation en bins et calcul du taux de 1
                df["_bin"] = pd.qcut(df[col], q=nb_bins, duplicates="drop")
                taux_1 = df.groupby("_bin", observed=False)[target].mean().reset_index()
                sns.barplot(x="_bin", y=target, data=taux_1, palette="viridis", hue="_bin", legend=False)
                plt.ylabel("Taux de 1")
                plt.title(f"Taux de 1 de {target} par {col} (bins)")
                plt.xticks(rotation=45)
                del df["_bin"]
            else:
                # Scatterplot classique
                sns.scatterplot(x=df[col], y=df[target], alpha=0.7)
                plt.title(f"{col} vs {target}")
                plt.xlabel(col)
                plt.ylabel(target)

        plt.tight_layout()
        plt.show()

    # --- Variables qualitatives et quantitatives discrètes ---
    if len(quant_discr_cat) > 0:
        print("\n--- Variables qualitatives / quantitatives discrètes ---")
        n = len(quant_discr_cat)
        n_cols = 2
        n_rows = (n + 1) // n_cols

        plt.figure(figsize=(n_cols * 6, n_rows * 4))

        for i, var in enumerate(quant_discr_cat):
            plt.subplot(n_rows, n_cols, i + 1)

            if cible_binaire:
                # Taux de 1
                sns.barplot(
                    x=var, y=target, data=df,
                    estimator="mean", hue=var, palette="viridis", legend=False
                )
                plt.ylabel("Taux de 1")
                plt.title(f"Taux de 1 de {target} par {var}")
            else:
                sns.barplot(
                    x=var, y=target, data=df,
                    estimator="mean", hue=var, palette="viridis", legend=False
                )
                plt.ylabel("Moyenne cible")
                plt.title(f"{target} par {var}")

            plt.xlabel(var)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
    else:
        print("Aucune variable qualitative ou discrète détectée.")
