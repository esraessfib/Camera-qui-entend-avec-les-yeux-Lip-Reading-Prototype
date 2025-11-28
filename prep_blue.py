import os
import cv2
import pandas as pd

# ============================
#   1. Configuration
# ============================
DATASET_PATH = r"...\projet\LipNet\data\blue\videos_blue"   # Dossier des vidéos contenant "blue"
OUTPUT_PATH = r"...\projet\LipNet\data\blue\prepared_blue" # Dossier de sortie

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================
#   2. Fonctions utilitairesa
# ============================
def read_align_file(path):
    """Lit un fichier .align et retourne le texte complet."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    words = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            _, _, word = parts
            if word != "sil":
                words.append(word)
    return " ".join(words)

def extract_frames(video_path, output_dir, every_n_frames=2):
    """Extrait une image sur every_n_frames et la sauvegarde."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            frame_file = os.path.join(output_dir, f"frame_{count:03d}.jpg")
            cv2.imwrite(frame_file, frame)
            count += 1
        frame_idx += 1
    cap.release()
    return count

# ============================
#   3. Boucle principale
# ============================
metadata = []

# Récupère toutes les vidéos .mpg du dossier blue
speakers = [d for d in os.listdir(DATASET_PATH) if d.startswith("s") and os.path.isdir(os.path.join(DATASET_PATH, d))]

for spk in speakers:
    speaker_dir = os.path.join(DATASET_PATH, spk)
    align_dir = os.path.join(speaker_dir, "align")
    
    videos = [f for f in os.listdir(speaker_dir) if f.endswith(".mpg")]

    for idx, video_file in enumerate(videos):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(speaker_dir, video_file)
        align_path = os.path.join(align_dir, video_name + ".align")

        if not os.path.exists(align_path):
            print(f"[!] Alignement manquant pour {video_file}, ignoré.")
            continue

        sentence = read_align_file(align_path)
        output_frames_dir = os.path.join(OUTPUT_PATH, f"{spk}_{video_name}")

        frame_count = extract_frames(video_path, output_frames_dir)
        metadata.append({
            "speaker": spk,
            "video_name": video_name,
            "frames_path": output_frames_dir,
            "num_frames": frame_count,
            "text": sentence
        })

        if (idx + 1) % 50 == 0:
            print(f"{idx + 1}/{len(videos)} vidéos traitées pour {spk}...")


# ============================
#   4. Sauvegarde du CSV
# ============================
df = pd.DataFrame(metadata)
csv_path = os.path.join(OUTPUT_PATH, "metadata_blue.csv")
df.to_csv(csv_path, index=False)

print("\n Préparation terminée !")
print(f" Dossier des frames : {OUTPUT_PATH}")
print(f" Fichier CSV : {csv_path}")
print(df.head())