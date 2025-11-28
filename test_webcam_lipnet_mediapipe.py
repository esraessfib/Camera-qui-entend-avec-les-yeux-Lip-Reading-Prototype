import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from collections import deque
import mediapipe as mp

# ===============================
# CONFIG
# ===============================
MODEL_PATH = r"F:\Documents\ensit\3ETA\traitement_image\projet\LipNet\models\lipnet_realtime.pt"
IMG_H, IMG_W = 50, 100
SEQ_LEN = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB = ['-', ' '] + [chr(i) for i in range(65, 91)]
idx2char = {i: c for i, c in enumerate(VOCAB)}

# ===============================
# FONCTIONS UTILES
# ===============================
def decode_ctc(output_tensor):
    pred = output_tensor.argmax(dim=2)  # (B, T)
    decoded = []
    for p in pred:
        prev = None
        word = ""
        for c in p.cpu().numpy():
            if c != 0 and c != prev:
                word += idx2char.get(c, '')
            prev = c
        decoded.append(word.strip())
    return decoded[0]

# ===============================
# CHARGEMENT DU MODÈLE
# ===============================
print("Chargement du modèle TorchScript...")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()
print(" Modèle chargé sur :", DEVICE)

# ===============================
# MEDIA PIPE POUR DÉTECTION LÈVRES
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Impossible d’ouvrir la webcam")

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

buffer = deque(maxlen=SEQ_LEN)
last_prediction = ""
cooldown = 0

print(" Appuyez sur 'q' pour quitter")

# ===============================
# BOUCLE TEMPS RÉEL
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lip_points = [face_landmarks.landmark[i] for i in range(61, 88)]
            xs = [int(p.x * w) for p in lip_points]
            ys = [int(p.y * h) for p in lip_points]

            x_min, x_max = max(0, min(xs) - 10), min(w, max(xs) + 10)
            y_min, y_max = max(0, min(ys) - 10), min(h, max(ys) + 10)
            lip_roi = frame[y_min:y_max, x_min:x_max]

            if lip_roi.size == 0:
                continue

            roi_rgb = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2RGB)
            roi_tensor = transform(roi_rgb)
            buffer.append(roi_tensor)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Prédiction quand le buffer est plein
            if len(buffer) == SEQ_LEN and cooldown == 0:
                seq_tensor = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(seq_tensor)
                decoded = decode_ctc(output)
                if decoded and decoded != last_prediction:
                    last_prediction = decoded
                    cooldown = 20
                    print(f" Mot détecté : {decoded}")

    if cooldown > 0:
        cooldown -= 1

    cv2.putText(frame, f"Prediction: {last_prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("LipNet - Webcam (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Capture terminée.")
