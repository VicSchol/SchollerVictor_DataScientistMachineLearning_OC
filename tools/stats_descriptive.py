import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, fisher_exact
import itertools

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

def analyse_graphique(df: pd.DataFrame, target: str, seuil_modalites: int = 15, nb_bins: int = 10, alpha: float = 0.05):
    """
    Analyse graphique complète :
    - Quantitatives continues : scatterplots ou bins pour cible binaire
    - Qualitatives / discrètes : barplots avec tests globaux et post-hoc annotés sur le graphe
    """
    cible_binaire = df[target].nunique() == 2

    # Séparation variables
    quant_cont = []
    quant_discr_cat = []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= seuil_modalites:
                quant_discr_cat.append(col)
            else:
                quant_cont.append(col)
        else:
            quant_discr_cat.append(col)

    # --- Variables quantitatives continues ---
    if quant_cont:
        n = len(quant_cont)
        n_cols = 2
        n_rows = (n + 1) // n_cols
        plt.figure(figsize=(n_cols*6, n_rows*4))

        for i, col in enumerate(quant_cont):
            plt.subplot(n_rows, n_cols, i+1)
            if cible_binaire:
                df["_bin"] = pd.qcut(df[col], q=nb_bins, duplicates="drop")
                taux_1 = df.groupby("_bin")[target].mean().reset_index()
                sns.barplot(x="_bin", y=target, data=taux_1, palette="viridis", hue="_bin", legend=False)
                plt.ylabel("Taux de 1")
                plt.title(f"{target} par {col} (bins)")
                plt.xticks(rotation=45)
                del df["_bin"]
            else:
                sns.scatterplot(x=df[col], y=df[target], alpha=0.7)
                plt.xlabel(col)
                plt.ylabel(target)
                plt.title(f"{col} vs {target}")

        plt.tight_layout()
        plt.show()

    # --- Variables qualitatives / discrètes ---
    if quant_discr_cat:
        n = len(quant_discr_cat)
        n_cols = 2
        n_rows = (n + 1) // n_cols
        plt.figure(figsize=(n_cols*6, n_rows*4))

        for i, var in enumerate(quant_discr_cat):
            plt.subplot(n_rows, n_cols, i+1)
            sns.barplot(x=var, y=target, data=df, estimator="mean", ci=None, palette="viridis")

            modalities = df[var].dropna().unique()
            y_max = df.groupby(var)[target].mean().max() * 1.1

            # Test global
            if cible_binaire:
                tab = pd.crosstab(df[var], df[target])
                if tab.shape == (2,2):
                    _, p_global = fisher_exact(tab)
                    test_name = "Fisher exact"
                else:
                    _, p_global, _, _ = chi2_contingency(tab)
                    test_name = "Chi²"
            else:
                groupes = [df[df[var]==m][target].dropna() for m in modalities]
                if len(groupes) == 2:
                    _, p_global = mannwhitneyu(groupes[0], groupes[1])
                    test_name = "Mann-Whitney U"
                else:
                    _, p_global = kruskal(*groupes)
                    test_name = "Kruskal-Wallis"

            # Post-hoc si global significatif
            if p_global < alpha and len(modalities) > 1:
                pairs = list(itertools.combinations(modalities, 2))
                for j, (g1, g2) in enumerate(pairs):
                    x1 = df[df[var]==g1][target].dropna()
                    x2 = df[df[var]==g2][target].dropna()
                    if len(x1) > 0 and len(x2) > 0:
                        if cible_binaire:
                            sub_tab = pd.crosstab(df[df[var].isin([g1,g2])][var], df[df[var].isin([g1,g2])][target])
                            _, p_pair = fisher_exact(sub_tab)[:2]
                        else:
                            _, p_pair = mannwhitneyu(x1, x2)
                        # Astérisques selon p-value
                        if p_pair < 0.001:
                            sig = '***'
                        elif p_pair < 0.01:
                            sig = '**'
                        elif p_pair < 0.05:
                            sig = '*'
                        else:
                            sig = ''
                        if sig:
                            x_coords = [list(modalities).index(g1), list(modalities).index(g2)]
                            y = y_max + j*0.05*y_max
                            plt.plot(x_coords, [y,y], color='black')
                            plt.text((x_coords[0]+x_coords[1])/2, y, sig, ha='center', va='bottom')

            plt.title(f"{var} (p global={p_global:.3f})")
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
