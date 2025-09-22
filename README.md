# 👀🎤 Caméra qui entend avec les yeux – Lip-Reading Prototype

## 🚀 Présentation

Ce projet vise à développer une **caméra intelligente** capable de reconnaître en **temps réel** quelques mots ou phrases simples à partir des **mouvements labiaux**, sans utiliser de microphone.
Le prototype repose sur un vocabulaire restreint (5 à 10 mots de base, ex : *yes, no, help, stop, go*) et sera déployé sur **Jetson Nano** ou **Jetson Xavier**.

Impact :

* 💬 Nouvelle solution de communication silencieuse.
* ♿ Accessibilité améliorée pour les personnes sourdes ou muettes.
* 🔬 Cas d’usage en sécurité, assistance et interfaces sans son.

---

## 🎯 Objectifs du projet

* Reconnaissance de **5–10 mots en anglais** via lecture labiale.
* Démonstration en temps réel (≥10 fps) sur Jetson Nano/Xavier.
* Interface simple affichant la vidéo + texte reconnu.
* Option : synthèse vocale (Text-to-Speech) pour convertir le texte reconnu en voix.

---

## ⏳ Contraintes & Faisabilité


  * Tache 1 : Acquisition & traitement vidéo.
  * Tache 2 : Modèle IA (entraînement).
  * Tache 3 : Déploiement Jetson + interface.
* **Matériel** : caméra USB + Jetson Nano 2Go.
* **Limites** :
  * Vocabulaire restreint.
  * Classification de mots courts (pas de phrases continues).
* **Dataset** :

  * Public : *GRID, LRW, LRS2*.
  * Interne : \~100 vidéos/mot/personne → \~1500 exemples pour 5 mots.

---

## 🛠️ Étapes techniques & Outils

### 1️⃣ Acquisition & Prétraitement vidéo

* Détection du visage et extraction de la région labiale.
* Normalisation des séquences (64×64 ou 112×112 px).
* **Outils** : OpenCV, Mediapipe (landmarks), Dlib (optionnel).

### 2️⃣ Constitution du dataset

* **Option A :** utilisation de datasets publics (*GRID, LRW*).
* **Option B :** enregistrement interne de \~1500 vidéos labiales.
* **Outils** : Python + OpenCV pour découpe et labellisation.

### 3️⃣ Modèle IA – Lip-Reading

* Objectif : classifier une séquence vidéo en un mot.
* Modèles envisagés :
  * CNN + RNN (VGG + LSTM).
  * 3D-CNN (C3D, I3D).
  * Transformer léger (LipNet simplifié).
* **Frameworks** : PyTorch (principal), TensorFlow (option).

### 4️⃣ Entraînement & Validation

* Entraînement sur PC avec GPU (RTX recommandé).
* Suivi avec TensorBoard ou Weights & Biases.
* Objectif : ≥80% précision sur vocabulaire restreint.

### 5️⃣ Déploiement temps réel

* Conversion modèle PyTorch → ONNX → TensorRT.
* Optimisation pour Jetson (10–15 fps).
* Interface : Python + OpenCV (vidéo + texte détecté).
* Option : TTS pour restitution vocale.

---

## 📊 Architecture du système

```
   [Caméra USB/CSI]
          │
          ▼
   OpenCV + Mediapipe
   (Extraction lèvres)
          │
          ▼
   PyTorch Model (CNN+RNN)
          │
          ▼
   Déploiement Jetson (TensorRT)
          │
    ┌─────────────┬─────────────┐
    ▼             ▼             ▼
 Affichage      Node-RED      TTS (option)
  Vidéo+Texte   Dashboard     Voix artificielle
```

---

## 📦 Logiciels & Librairies

* **Traitement vidéo** : OpenCV, Mediapipe, Dlib.
* **IA** : PyTorch, TorchVision, ONNX, TensorRT.
* **Suivi entraînement** : TensorBoard / Weights & Biases.
* **Interface** : Python (OpenCV GUI).
* **Optionnel** : gTTS ou pyttsx3 pour TTS.

---



## 🔮 Améliorations futures

* Extension du vocabulaire (20–50 mots).
* Support multi-langues.
* Détection continue de phrases (lip-to-text).
* Intégration avec applications mobiles.

---

