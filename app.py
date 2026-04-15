import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import os

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
    # Modelos
    ae = Autoencoder(canales_latentes=8)
    ae = load_weights_safe(ae, 'autoencoder.pth')
    cnn = EfficientNetFingerprint()
    cnn = load_weights_safe(cnn, 'cnn_fingerprint.pth')
    
    # Base de Datos (con carga robusta per debug)
    db_path = 'database_impronte.pt'
    db = None
    
    if os.path.exists(db_path):
        try:
            # Usamos weights_only=False por compatibilidad con archivos .pt de diccionarios
            db = torch.load(db_path, map_location='cpu', weights_only=False)
        except Exception as e:
            st.error(f"❌ Error técnico al leer la base de datos: {e}")
    else:
        st.error(f"❌ CRÍTICO: El archivo '{db_path}' non si trova nella cartella principale di GitHub.")
        
    return ae, cnn, db

def get_fingerprint(audio_file, ae, cnn):
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img).unsqueeze(0)
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    return fp.numpy().flatten(), S_db

# ==========================================================
# 3. INTERFAZ STREAMLIT
# ==========================================================

st.set_page_config(page_title="Copyright Detector", page_icon="🎵")
st.title("🎵 Detector de Copyright y Plagio")
st.markdown("### Taller de Tecnomatemática")

try:
    ae, cnn, db_huellas = load_assets()

    if db_huellas is None:
        st.warning("⚠️ El sistema no puede funcionar sin la base de datos cargada.")
        st.info("Asegúrate de que 'database_impronte.pt' esté en la raíz de tu repositorio GitHub.")
        st.stop()

    st.success(f"Sistema listo. Base de datos: **{len(db_huellas)}** canciones.")
    
    st.divider()
    
    # OPCIONES DE ENTRADA
    st.subheader("🎤 Analizar Audio")
    tab1, tab2 = st.tabs(["🎙️ Escuchar (Shazam)", "📁 Cargar Archivo"])
    
    with tab1:
        audio_input = st.audio_input("Graba 10 segundos de música")
        
    with tab2:
        audio_file = st.file_uploader("Sube un archivo mp3/wav", type=["mp3", "wav"])

    u_audio = audio_input if audio_input else audio_file

    if u_audio:
        if st.button("🚀 VERIFICAR AHORA", use_container_width=True):
            with st.spinner("Procesando huella acústica..."):
                fp_sospechoso, s_sospechoso = get_fingerprint(u_audio, ae, cnn)
                
                mejor_sim = -1
                match_name = ""
                
                for nombre, fp_ref in db_huellas.items():
                    sim = np.dot(fp_sospechoso, fp_ref) * 100
                    if sim > mejor_sim:
                        mejor_sim = sim
                        match_name = nombre

                st.divider()
                st.header("Veredicto")

                if mejor_sim >= 98:
                    st.success("✅ **ES LA CANCIÓN ORIGINAL**")
                    st.write(f"Coincidencia exacta con: **{match_name}**")
                elif mejor_sim >= 80:
                    st.warning("🚨 **SÍ, ES UNA COPIA / PLAGIO**")
                    st.write(f"Se detectó una alta similitud con: **{match_name}**")
                    st.caption(f"Confianza: {mejor_sim:.2f}%")
                else:
                    st.error("❌ **NO ES UNA COPIA**")
                    st.write("No se encontraron coincidencias en el dataset.")

                with st.expander("Ver detalles técnicos"):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    librosa.display.specshow(s_sospechoso, ax=ax, cmap='magma')
                    st.pyplot(fig)

except Exception as e:
    st.error(f"Ocurrió un error inesperado: {e}")