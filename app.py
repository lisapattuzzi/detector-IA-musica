import streamlit as st
import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# --- 1. ARQUITECTURAS (Sin cambios) ---
class Autoencoder(nn.Module):
    def __init__(self, canales_latentes=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, canales_latentes, 3, stride=2, padding=1), nn.Tanh()
        )
    def forward(self, x): return self.encoder(x)

class EfficientNetFingerprint(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        c_orig = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(8, c_orig.out_channels, 3, stride=c_orig.stride, padding=c_orig.padding, bias=False)
        self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier[1].in_features, dim))
    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return F.normalize(self.backbone(x), p=2, dim=1)

# --- 2. TU FUNCIÓN EXACTA (Adaptada a Streamlit) ---
def obtener_huella_y_espectrograma(audio_bytes, ae, cnn):
    # Usamos io.BytesIO para leer el audio subido
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    # --- INICIO DE TU CÓDIGO EXACTO ---
    D = np.abs(librosa.stft(y))**2  # Compute the Short-Time Fourier Transform (STFT)
    S = librosa.power_to_db(D, ref=np.max)  # Convert to logarithm scale

    # Create the spectrogram image
    fig = plt.figure(figsize=(4, 4))
    librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log")
    plt.axis("off")
    # --- FIN DE TU CÓDIGO EXACTO ---

    # En lugar de save(output_path), guardamos en un buffer de memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Procesamiento de la imagen para la IA (manteniendo la coherencia)
    img_pil = Image.open(buf)
    # Mostramos a color en la App, pero convertimos a 'L' (gris) para el Autoencoder
    img_for_ai = img_pil.convert('L').resize((256, 256))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_img = transform(img_for_ai).unsqueeze(0)

    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    
    return fp.numpy().flatten(), img_pil

# --- 3. CARGA Y LÓGICA DE LA APP (En Español) ---
@st.cache_resource
def load_assets():
    ae = Autoencoder(8)
    cnn = EfficientNetFingerprint(128)
    ae.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'), strict=False)
    cnn.load_state_dict(torch.load('cnn_fingerprint.pth', map_location='cpu'), strict=False)
    ae.eval()
    cnn.eval()
    db = torch.load('database_impronte.pt', map_location='cpu', weights_only=False)
    return ae, cnn, db

st.title("🎵 Analizador de Autenticidad Musical")
ae, cnn, db_huellas = load_assets()

archivo = st.file_uploader("Sube un archivo de audio (.mp3 o .wav)", type=["mp3", "wav"])

if archivo:
    audio_content = archivo.read()
    st.audio(audio_content)
    
    if st.button("Analizar"):
        with st.spinner("Procesando espectrograma..."):
            huella_input, img_color = obtener_huella_y_espectrograma(audio_content, ae, cnn)
            
            # Aquí verás el espectrograma idéntico al que me mostraste
            st.image(img_color, caption="Espectrograma generado (Copia exacta del dataset)", width=350)

            # Comparación
            max_sim = -1
            mejor_match = ""
            # --- AÑADE ESTAS LÍNEAS AQUÍ PARA DEBUGEAR ---
            st.subheader("🔍 Información de Control (Debug)")
            st.write(f"Total de canciones en DB: {len(db_huellas)}")
            st.write("Primeros 5 nombres en la base de datos:", list(db_huellas.keys())[:5])
            # --------------------------------------------
            for nombre, huella_db in db_huellas.items():
                sim = np.dot(huella_input, huella_db)
                if sim > max_sim:
                    max_sim = sim
                    mejor_match = nombre

            # Lógica de respuesta
            st.subheader("Resultado:")
            if max_sim >= 0.95:
                st.success(f"✅ **La canción está presente en el dataset.**")
                st.write(f"Coincidencia: {mejor_match}")
            elif 0.70 <= max_sim < 0.95:
                st.warning(f"🤖 **Es una COVER generada por IA.**")
                st.write(f"Basada en: {mejor_match}")
            else:
                st.error(f"❌ **La canción no tiene nada que ver.**")
                st.write("No se encontraron similitudes significativas.")
            
            st.write(f"Similitud técnica: {max_sim:.4f}")