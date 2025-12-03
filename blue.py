import os
import shutil

# =============================
# Configuration
# =============================
DATASET_PATH = r"...\projet\LipNet\data\grid_corpus"
OUTPUT_PATH = r"...\projet\LipNet\data\videos_blue"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =============================
# Fonctions utilitaires
# =============================

def find_video_file(base_name, search_root):
    """Cherche un fichier vidéo correspondant à base_name dans search_root."""
    for root, _, files in os.walk(search_root):
        for f in files:
            if f.startswith(base_name) and f.endswith(".mpg"):
                return os.path.join(root, f)
    return None

# =============================
# Parcours principal
# =============================

count = 0

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".align"):
            align_path = os.path.join(root, file)
            base_name = os.path.splitext(file)[0]

            with open(align_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "blue" in content.split():
                # Chercher le fichier vidéo correspondant
                video_path = find_video_file(base_name, DATASET_PATH)

                # Copier le .align
                shutil.copy(align_path, os.path.join(OUTPUT_PATH, file))

                if video_path and os.path.exists(video_path):
                    shutil.copy(video_path, os.path.join(OUTPUT_PATH, os.path.basename(video_path)))
                    print(f" Copié : {base_name}.align + {base_name}.mpg")
                else:
                    print(f" Vidéo introuvable pour {base_name}")

                count += 1

print(f"\n Terminé ! {count} fichiers .align contenant 'blue' ont été trouvés.")
print(f" Tous les fichiers copiés dans : {OUTPUT_PATH}")
