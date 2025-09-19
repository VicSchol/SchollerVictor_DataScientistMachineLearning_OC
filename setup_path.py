# setup_path.py
import sys
import os

try:
    # script classique
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # notebook ou interactive window
    ROOT_DIR = os.getcwd()

# Ajouter la racine et tous les sous-dossiers au sys.path
for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
    if dirpath not in sys.path:
        sys.path.append(dirpath)