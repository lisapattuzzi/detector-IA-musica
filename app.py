import streamlit as st
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
import os
import io

# ==========================================================
# 1. DEFINICIÓN DE ARQUITECTURAS
# ==========================================================

class Autoencoder(nn.Module):
    def __init__(self, canales_latentes=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, canales_latentes, 3, stride=2, padding=1), nn.Tanh()
        )
    def forward(self, x): 
        return self.encoder(x)

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

# ==========================================================
# 2. CARGA DE MODELOS (Caché)
# ==========================================================

@st.cache_resource
def load_assets():
    ae = Autoencoder(canales_latentes=8)
    cnn = EfficientNetFingerprint()
    db = None
    
    if os.path.exists('autoencoder.pth'):
        state_dict = torch.load('autoencoder.pth', map_location='cpu')
        ae.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
        ae.eval()
        
    if os.path.exists('cnn_fingerprint.pth'):
        state_dict = torch.load('cnn_fingerprint.pth', map_location='cpu')
        cnn.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
        cnn.eval()
        
    if os.path.exists('database_impronte.pt'):
        db = torch.load('database_impronte.pt', map_location='cpu', weights_only=False)
        
    return ae, cnn, db

# ==========================================================
# 3. EXTRACCIÓN EXACTA (Basada en el código original)
# ==========================================================

def process_audio_exact(audio_file, ae, cnn):
    """Genera la imagen usando LA MISMA RECETA que el dataset de entrenamiento"""
    # 1. Cargar el audio completo con su Sample Rate original
    y, sr = librosa.load(audio_file, sr=None)
    
    # 2. Calcular STFT (No Mel-Spectrogram)
    D = np.abs(librosa.stft(y))**2
    S = librosa.power_to_db(D, ref=np.max)
    
    # 3. Crear la imagen usando los mismos parámetros (4x4, log, sin ejes)
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log")
    plt.axis("off")
    
    # 4. Guardar en memoria exactamente igual
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    
    # 5. Transformar para la Red Neuronal
    img_pil = Image.open(buf)
    img_gray = img_pil.convert('L').resize((256, 256))
    
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img_gray).unsqueeze(0)
    
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
        
    return fp.numpy().flatten(), img_pil

# ==========================================================
# 4. INTERFAZ DE USUARIO LIMPIA
# ==========================================================

st.set_page_config(page_title="Copyright Detector", page_icon="🎵", layout="centered")

st.title("🎵 Escáner de Plagio y Copyright")
st.markdown("Sube una canción. El sistema analizará la estructura acústica completa de la pista (STFT).")

ae, cnn, db_huellas = load_assets()

if db_huellas is None:
    st.error("⚠️ Base de datos no encontrada. Falta 'database_impronte.pt'.")
    st.stop()

audio_file = st.file_uploader("📁 Selecciona un archivo MP3 o WAV", type=["mp3", "wav"])

if audio_file:
    if st.button("🚀 INICIAR ANÁLISIS", use_container_width=True):
        
        with st.spinner("Procesando la canción completa (Esto puede tardar unos segundos)..."):
            try:
                # Extraer huella
                fp_sospechoso, img_visual = process_audio_exact(audio_file, ae, cnn)
                
                # Comparar con DB
                resultados = []
                for nombre, fp_ref in db_huellas.items():
                    sim = np.dot(fp_sospechoso, fp_ref) * 100
                    resultados.append((nombre, sim))
                
                resultados.sort(key=lambda x: x[1], reverse=True)
                match_name, mejor_sim = resultados[0]
                
                # Veredicto
                st.divider()
                st.header("Veredicto Final")

                if mejor_sim >= 90:
                    st.success(f"✅ **COINCIDENCIA EXACTA CONFIRMADA**\n\nIdentificada como: **{match_name}**")
                elif mejor_sim >= 60:
                    st.warning(f"🚨 **ALERTA DE POSIBLE PLAGIO**\n\nAlta correlación con: **{match_name}**")
                else:
                    st.error("❌ **AUDIO LIMPIO**\n\nNo se encontraron copias en la base de datos.")

                # Detalles
                with st.expander("📊 Ver Detalles Técnicos (Top 3)"):
                    st.write(f"Porcentaje de confianza máximo: **{mejor_sim:.2f}%**")
                    for i, (nombre, sim) in enumerate(resultados[:3]):
                        st.write(f"**{i+1}.** {nombre} ➔ `{sim:.2f}%`")
                    
                    st.markdown("### 🎛️ Espectrograma STFT Generado")
                    st.image(img_visual, caption="Imagen procesada por la Red Neuronal", use_column_width=True)

            except Exception as e:
                st.error(f"Error procesando el audio: {e}")