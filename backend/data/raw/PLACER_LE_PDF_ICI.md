# Emplacement du corpus

Copier ici la these, sous le nom exact :

    these_vagues_froid.pdf

Commande Windows (PowerShell), depuis la racine du projet :

    Copy-Item "C:\chemin\vers\01_DATA_These_Vagues_Froid.pdf" "backend\data\raw\these_vagues_froid.pdf"

Le PDF original ne doit JAMAIS etre modifie : ce dossier est en lecture seule pour
tout le pipeline. Il n'est pas versionne (voir .gitignore, 47 Mo).
