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

# --- 1. ARQUITECTURAS DE LAS REDES ---
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

# --- 2. FUNCIÓN DE GENERACIÓN (IDÉNTICA A TU CÓDIGO DE COLAB) ---
def generar_huella_exacta(audio_bytes, ae, cnn):
    # Cargar audio
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    # STFT y escala Logarítmica (Exactamente como tu función save_spectrogram)
    D = np.abs(librosa.stft(y))**2
    S = librosa.power_to_db(D, ref=np.max)

    # Crear la imagen del espectrograma (figsize 4,4 como pediste)
    fig = plt.figure(figsize=(4, 4))
    librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log")
    plt.axis("off")
    
    # Guardar en buffer para procesar
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Pre-procesamiento para la IA
    img = Image.open(buf).convert('L').resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_img = transform(img).unsqueeze(0)

    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    
    return fp.numpy().flatten(), img

# --- 3. CARGA DE MODELOS Y DATASET ---
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

# --- 4. INTERFAZ DE USUARIO ---
st.set_page_config(page_title="Detector de IA Musical", page_icon="🎵")
st.title("🎵 Analizador de Autenticidad Musical")

ae, cnn, db_huellas = load_assets()

archivo = st.file_uploader("Sube una canción (.mp3 o .wav)", type=["mp3", "wav"])

if archivo:
    st.audio(archivo)
    
    if st.button("Analizar Canción"):
        with st.spinner("Comparando con el dataset..."):
            huella_input, img_espectrograma = generar_huella_exacta(archivo.read(), ae, cnn)
            
            # Mostrar espectrograma generado
            st.image(img_espectrograma, caption="Espectrograma Generado (Log-scale STFT)", width=300)

            # Lógica de comparación
            max_similitud = -1
            mejor_match = ""

            for nombre, huella_db in db_huellas.items():
                similitud = np.dot(huella_input, huella_db)
                if similitud > max_similitud:
                    max_similitud = similitud
                    mejor_match = nombre

            # --- 5. RESULTADOS SEGÚN TUS 3 PUNTOS ---
            st.subheader("Resultado del Análisis:")
            
            if max_similitud >= 0.95:
                st.success(f"✅ **La canción está presente en el dataset.**")
                st.info(f"Coincidencia exacta con: **{mejor_match}**")
            
            elif 0.70 <= max_similitud < 0.95:
                st.warning(f"🤖 **Es una COVER generada por IA.**")
                st.write(f"Se detectó una base similar a: *{mejor_match}*, pero con variaciones artificiales.")
            
            else:
                st.error(f"❌ **La canción no tiene nada que ver.**")
                st.write("No se encontraron patrones similares en la base de datos.")

            st.progress(float(max_similitud) if max_similitud > 0 else 0.0)
            st.write(f"Nivel de similitud técnica: {max_similitud:.4f}")