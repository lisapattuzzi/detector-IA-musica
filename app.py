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

# --- 1. ARQUITECTURAS DE LAS REDES (Deben coincidir con tu entrenamiento) ---
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
        # Ajuste para recibir los 8 canales del Autoencoder
        self.backbone.features[0][0] = nn.Conv2d(8, c_orig.out_channels, 3, stride=c_orig.stride, padding=c_orig.padding, bias=False)
        self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier[1].in_features, dim))
    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return F.normalize(self.backbone(x), p=2, dim=1)

# --- 2. FUNCIÓN DE PROCESAMIENTO (IDÉNTICA A TU COLAB) ---
def estrai_impronta(audio_bytes, ae, cnn):
    # Cargar audio desde los bytes subidos
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    # Calcular espectrograma STFT
    D = np.abs(librosa.stft(y))**2
    S = librosa.power_to_db(D, ref=np.max)

    # Generar imagen con Matplotlib (Igual que en el dataset)
    fig = plt.figure(figsize=(4, 4), dpi=100)
    librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log")
    plt.axis("off")

    # Guardar en buffer de memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Procesar imagen para la red (256x256, Grayscale)
    img = Image.open(buf).convert('L').resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_img = transform(img).unsqueeze(0)

    # Extraer vector de características (Fingerprint)
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    
    return fp.numpy().flatten(), img

# --- 3. CARGA DE RECURSOS (Modelos y Base de Datos) ---
@st.cache_resource
def load_assets():
    ae = Autoencoder(8)
    cnn = EfficientNetFingerprint(128)
    
    # Cargar pesos (Asegúrate de que estos nombres coincidan con tus archivos en GitHub)
    ae.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'), strict=False)
    cnn.load_state_dict(torch.load('cnn_fingerprint.pth', map_location='cpu'), strict=False)
    ae.eval()
    cnn.eval()
    
    # Cargar Base de Datos con el parámetro de seguridad weights_only=False
    db = {}
    if os.path.exists('database_impronte.pt'):
        db = torch.load('database_impronte.pt', map_location='cpu', weights_only=False)
    
    return ae, cnn, db

# --- 4. INTERFAZ DE STREAMLIT ---
st.title("🎵 Detector de IA en Música")
st.write("Sube un fragmento de audio para verificar si coincide con nuestra base de datos.")

ae, cnn, db_huellas = load_assets()

archivo_subido = st.file_uploader("Elige un archivo de audio (mp3, wav)", type=["mp3", "wav"])

if archivo_subido is not None:
    st.audio(archivo_subido, format='audio/mp3')
    
    if st.button("Analizar Audio"):
        with st.spinner("Generando espectrograma y comparando..."):
            # Extraer huella del audio subido
            huella_audio, espectrograma_img = estrai_impronta(archivo_subido.read(), ae, cnn)
            
            # Mostrar el espectrograma generado (para control visual)
            st.image(espectrograma_img, caption="Espectrograma analizado", use_container_width=True)
            
            if not db_huellas:
                st.error("La base de datos está vacía o no se pudo cargar.")
            else:
                # Comparación por Producto Punto (Similitud de Coseno)
                mejores_coincidencias = []
                for nombre, huella_db in db_huellas.items():
                    similitud = np.dot(huella_audio, huella_db)
                    mejores_coincidencias.append((nombre, similitud))
                
                # Ordenar por similitud
                mejores_coincidencias.sort(key=lambda x: x[1], reverse=True)
                
                # Mostrar resultados
                top_nombre, top_sim = mejores_coincidencias[0]
                
                if top_sim > 0.85:  # Umbral de confianza
                    st.success(f"✅ ¡Coincidencia encontrada! Canción: **{top_nombre}**")
                    st.write(f"Nivel de confianza: {top_sim:.2%}")
                else:
                    st.warning("⚠️ No se encontró una coincidencia exacta en la base de datos.")
                    st.write(f"La canción más parecida es: **{top_nombre}** (Similitud: {top_sim:.2%})")

st.divider()
st.caption(f"Base de datos actual: {len(db_huellas)} canciones registradas.")