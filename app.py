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
        # Adaptamos la entrada para los canales latentes
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
    # Cargar Modelos
    ae = Autoencoder(canales_latentes=8)
    ae = load_weights_safe(ae, 'autoencoder.pth')
    cnn = EfficientNetFingerprint()
    cnn = load_weights_safe(cnn, 'cnn_fingerprint.pth')
    
    # Cargar Base de Datos
    db_path = 'database_impronte.pt'
    db = None
    
    if os.path.exists(db_path):
        try:
            db = torch.load(db_path, map_location='cpu', weights_only=False)
        except Exception as e:
            st.error(f"❌ Error al leer la base de datos: {e}")
    else:
        st.error(f"❌ CRÍTICO: El archivo '{db_path}' no se encuentra en el repositorio.")
        
    return ae, cnn, db

# ==========================================================
# 3. EXTRACCIÓN DE LA HUELLA ACÚSTICA
# ==========================================================

def get_fingerprint(audio_file, ae, cnn):
    # Leemos solo los primeros 10 segundos
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Transformamos el espectrograma en una imagen compatible con la red
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img).unsqueeze(0)
    
    # Pasamos la imagen por las redes para obtener el vector de 128D
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
        
    return fp.numpy().flatten(), S_db

# ==========================================================
# 4. INTERFAZ GRÁFICA (Streamlit)
# ==========================================================

st.set_page_config(page_title="Copyright Detector", page_icon="🎵")
st.title("🎵 Detector de Copyright y Plagio")
st.markdown("### Taller de Tecnomatemática - Sistema de Audio Fingerprinting")

try:
    ae, cnn, db_huellas = load_assets()

    if db_huellas is None:
        st.warning("⚠️ El sistema no puede funcionar sin la base de datos.")
        st.stop()

    st.success(f"✅ Sistema en línea. Base de datos cargada: **{len(db_huellas)}** canciones.")
    st.divider()
    
    # SECCIÓN DE ENTRADA
    st.subheader("🎤 Analizar Pista de Audio")
    tab1, tab2 = st.tabs(["📁 Cargar Archivo MP3/WAV", "🎙️ Escuchar (Micrófono)"])
    
    with tab1:
        audio_file = st.file_uploader("Sube la canción a verificar", type=["mp3", "wav"])
    with tab2:
        st.info("Asegúrate de conceder permisos de micrófono al navegador.")
        audio_input = st.audio_input("Graba la música reproduciéndose")

    u_audio = audio_file if audio_file else audio_input

    if u_audio:
        if st.button("🚀 INICIAR ESCÁNER", use_container_width=True):
            with st.spinner("Extrayendo características topológicas del audio..."):
                
                fp_sospechoso, s_sospechoso = get_fingerprint(u_audio, ae, cnn)
                
                # Calcular similitud del coseno con TODA la base de datos
                resultados = []
                for nombre, fp_ref in db_huellas.items():
                    # Cálculo del producto punto (Similitud del Coseno)
                    sim = np.dot(fp_sospechoso, fp_ref) * 100
                    resultados.append((nombre, sim))

                # Ordenar de mayor a menor similitud
                resultados.sort(key=lambda x: x[1], reverse=True)
                
                # Tomar el mejor resultado
                match_name, mejor_sim = resultados[0]

                # ==========================================
                # VEREDICTO FINAL
                # ==========================================
                st.divider()
                st.header("Resultado del Análisis")

                # Umbrales ajustados para mayor precisión práctica
                if mejor_sim >= 95:
                    st.success("✅ **ES LA CANCIÓN ORIGINAL**")
                    st.write(f"Coincidencia exacta confirmada con: **{match_name}**")
                
                elif mejor_sim >= 60:
                    st.warning("🚨 **ALERTA DE PLAGIO / MODIFICACIÓN**")
                    st.write(f"Se detectó una alta correlación estructural con: **{match_name}**")
                    st.caption(f"El audio parece ser una versión modificada (pitch, tempo, ruido).")
                
                else:
                    st.error("❌ **NO ES UNA COPIA**")
                    st.write("La firma acústica es independiente. No hay coincidencias en el registro.")

                # ==========================================
                # DETALLES TÉCNICOS Y DEBUG (Para el profesor)
                # ==========================================
                with st.expander("⚙️ Ver detalles matemáticos y Top 3 Coincidencias"):
                    st.markdown("### 📊 Top 3 Similitud Vectorial")
                    for i, (nombre, sim) in enumerate(resultados[:3]):
                        st.write(f"**{i+1}.** {nombre} ➔ `{sim:.2f}%`")
                    
                    st.divider()
                    st.markdown("### 🎛️ Espectrograma Capturado (Transformada de Mel)")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    librosa.display.specshow(s_sospechoso, ax=ax, cmap='magma', y_axis='mel', x_axis='time')
                    ax.set_title("Firma Espectral ingresada")
                    plt.colorbar(format='%+2.0f dB')
                    st.pyplot(fig)

except Exception as e:
    st.error(f"Error crítico en la ejecución matemática: {e}")