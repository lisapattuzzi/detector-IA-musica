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
# 1. DEFINICIÓN DE ARQUITECTURAS (Redes Neuronales)
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
# 2. CARGA DE MODELOS Y BASE DE DATOS
# ==========================================================

def load_weights_safe(model, path):
    if not os.path.exists(path):
        st.error(f"⚠️ Archivo de pesos no encontrado: {path}")
        return model
    try:
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error cargando {path}: {e}")
        return model

@st.cache_resource
def load_assets():
    ae = Autoencoder(canales_latentes=8)
    ae = load_weights_safe(ae, 'autoencoder.pth')
    cnn = EfficientNetFingerprint()
    cnn = load_weights_safe(cnn, 'cnn_fingerprint.pth')
    
    db_path = 'database_impronte.pt'
    db = None
    if os.path.exists(db_path):
        try:
            db = torch.load(db_path, map_location='cpu', weights_only=False)
        except Exception as e:
            st.error(f"❌ Error al leer la base de datos: {e}")
    else:
        st.error(f"❌ CRÍTICO: El archivo '{db_path}' no está presente en el repositorio.")
        
    return ae, cnn, db

# ==========================================================
# 3. EXTRACCIÓN DE LA HUELLA (Corregido para Domain Matching)
# ==========================================================

def get_fingerprint(audio_file, ae, cnn):
    # 1. Cargamos solo los primeros 10 segundos
    y, sr = librosa.load(audio_file, duration=10)
    
    # 2. Parámetros IDÉNTICOS al dataset original (n_mels=128)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 3. Recreamos la imagen exactamente como en el dataset (Matplotlib + Magma)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.axis('off')
    librosa.display.specshow(S_db, sr=sr, ax=ax, cmap='magma')
    
    # Guardado en un buffer RAM para no escribir archivos en el disco duro
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    # 4. Transformación de la imagen para la red neuronal
    img = Image.open(buf).convert('L').resize((256, 256))
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img).unsqueeze(0)
    
    # 5. Generación del vector 128D
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
        
    return fp.numpy().flatten(), S_db

# ==========================================================
# 4. INTERFAZ STREAMLIT
# ==========================================================

st.set_page_config(page_title="Copyright Detector", page_icon="🎵")
st.title("🎵 Detector de Copyright y Plagio")
st.markdown("### Taller de Tecnomatemática - Sistema de Audio Fingerprinting")

try:
    ae, cnn, db_huellas = load_assets()

    if db_huellas is None:
        st.warning("⚠️ El sistema no puede funcionar sin la base de datos.")
        st.stop()

    st.success(f"✅ Sistema en línea. Canciones en la base de datos: **{len(db_huellas)}**")
    
    # MOSTRAR LISTA DE CANCIONES (Para depuración rápida)
    with st.expander("📚 Ver canciones cargadas en la base de datos"):
        st.write(list(db_huellas.keys()))

    st.divider()
    
    st.subheader("🎤 Analizar Pista de Audio")
    tab1, tab2 = st.tabs(["📁 Cargar Archivo MP3/WAV", "🎙️ Usar Micrófono"])
    
    with tab1:
        audio_file = st.file_uploader("Sube el archivo aquí", type=["mp3", "wav"])
    with tab2:
        audio_input = st.audio_input("Graba audio en tiempo real")

    u_audio = audio_file if audio_file else audio_input

    if u_audio:
        if st.button("🚀 INICIAR ESCÁNER", use_container_width=True):
            with st.spinner("Alineando espectrogramas y cálculo vectorial..."):
                
                fp_sospechoso, s_sospechoso = get_fingerprint(u_audio, ae, cnn)
                
                # Cálculo de similitud con toda la base de datos
                resultados = []
                for nombre, fp_ref in db_huellas.items():
                    sim = np.dot(fp_sospechoso, fp_ref) * 100
                    resultados.append((nombre, sim))

                resultados.sort(key=lambda x: x[1], reverse=True)
                match_name, mejor_sim = resultados[0]

                st.divider()
                st.header("Veredicto")

                if mejor_sim >= 95:
                    st.success(f"✅ **ES LA CANCIÓN ORIGINAL**\n\nCoincidencia con: **{match_name}**")
                elif mejor_sim >= 60:
                    st.warning(f"🚨 **ALERTA DE PLAGIO / MODIFICACIÓN**\n\nAlta correlación con: **{match_name}**")
                else:
                    st.error("❌ **NO ES UNA COPIA**\n\nNo se encontraron coincidencias en el dataset.")

                # SECCIÓN DE DEBUG (Fundamental para el profesor)
                with st.expander("⚙️ Detalles Técnicos (Top 3 Coincidencias)"):
                    for i, (nombre, sim) in enumerate(resultados[:3]):
                        st.write(f"**{i+1}.** {nombre} ➔ `{sim:.2f}%` de similitud")
                    
                    st.divider()
                    st.markdown("### 🎛️ Espectrograma generado (Modelo Magma)")
                    fig_debug, ax_debug = plt.subplots(figsize=(8, 3))
                    librosa.display.specshow(s_sospechoso, ax=ax_debug, cmap='magma', y_axis='mel', x_axis='time')
                    st.pyplot(fig_debug)

except Exception as e:
    st.error(f"Error crítico: {e}")