import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation(df, figsize=(12,10), annot=True, cmap='coolwarm', fontsize=12):
    """
    Calcule et affiche la matrice de corrélation pour les variables quantitatives
    avec meilleure lisibilité et moitié supérieure.
    
    Args:
        df (pd.DataFrame): le DataFrame à analyser
        figsize (tuple): taille du graphique
        annot (bool): afficher les valeurs dans la heatmap
        cmap (str): palette de couleurs pour la heatmap
        fontsize (int): taille des valeurs dans la heatmap
    """
    # Sélectionner uniquement les colonnes numériques
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculer la matrice de corrélation
    corr_matrix = df_numeric.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(15,9))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", mask=mask,
                annot_kws={"size": 8}, cmap='coolwarm', square=True,
                cbar_kws={"shrink": 0.8})

    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix
