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

# ==========================================================
# 1. DEFINICIÓN DE ARQUITECTURAS (Cerebro de la IA)
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
        # Adaptamos para recibir los 8 canales latentes del Autoencoder
        self.backbone.features[0][0] = nn.Conv2d(8, c_orig.out_channels, 3, stride=c_orig.stride, padding=c_orig.padding, bias=False)
        self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier[1].in_features, dim))
    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return F.normalize(self.backbone(x), p=2, dim=1)

# ==========================================================
# 2. FUNCIONES DE APOYO (Carga y Procesamiento)
# ==========================================================

def load_weights_safe(model, path):
    try:
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception:
        return model

@st.cache_resource
def load_assets():
    # Cargar modelos
    ae = Autoencoder(canales_latentes=8)
    ae = load_weights_safe(ae, 'autoencoder.pth')
    cnn = EfficientNetFingerprint()
    cnn = load_weights_safe(cnn, 'cnn_fingerprint.pth')
    # Cargar Base de Datos de huellas digitales
    try:
        db = torch.load('database_impronte.pt', map_location='cpu')
    except:
        db = None
    return ae, cnn, db

def get_fingerprint(audio_file, ae, cnn):
    # Cargar audio y generar Espectrograma de Mel
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Preprocesamiento de imagen
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img).unsqueeze(0)
    
    # Extracción del vector (Huella)
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    return fp.numpy().flatten(), S_db

# ==========================================================
# 3. INTERFAZ DE USUARIO (Streamlit)
# ==========================================================

st.set_page_config(page_title="Detector de Copyright Musical", page_icon="🔍")
st.title("🔍 Verificador de Plagio y Copyright")
st.markdown("### Taller de Tecnomatemática: Sistema 'Shazam' de Reconocimiento")

try:
    ae, cnn, db_huellas = load_assets()

    if db_huellas is None:
        st.error("❌ Falta el archivo 'database_impronte.pt'. Genéralo en Colab y súbelo.")
        st.stop()

    st.write(f"✅ Sistema activo. **{len(db_huellas)}** canciones en la base de datos.")
    
    # SECCIÓN DE ENTRADA DE AUDIO
    st.divider()
    st.subheader("🎤 Entrada de Audio")
    tab1, tab2 = st.tabs(["🎙️ Escuchar ahora (Micrófono)", "📁 Subir archivo"])
    
    with tab1:
        st.write("Haz clic en el icono para grabar un fragmento de la canción.")
        audio_grabado = st.audio_input("Grabación en vivo")
        
    with tab2:
        st.write("Selecciona un archivo de audio desde tu dispositivo.")
        audio_subido = st.file_uploader("Archivo de audio", type=["mp3", "wav"])

    # Selección de fuente
    u_audio = audio_grabado if audio_grabado else audio_subido

    if u_audio:
        if st.button("🚀 INICIAR VERIFICACIÓN", use_container_width=True):
            with st.spinner("Analizando señal acústica..."):
                # 1. Extraer huella del audio actual
                fp_sospechoso, s_sospechoso = get_fingerprint(u_audio, ae, cnn)
                
                mejor_similitud = -1
                cancion_match = ""
                
                # 2. Buscar en el registro (Algebra Lineal)
                for nombre, fp_ref in db_huellas.items():
                    sim = np.dot(fp_sospechoso, fp_ref) * 100
                    if sim > mejor_similitud:
                        mejor_similitud = sim
                        cancion_match = nombre

                # 3. Mostrar resultados según el umbral
                st.divider()
                st.header("Veredicto Final")

                if mejor_similitud >= 98:
                    st.success("✅ **ES LA CANCIÓN ORIGINAL**")
                    st.info(f"Identidad confirmada: **{cancion_match}**.")
                
                elif mejor_similitud >= 80:
                    st.warning("🚨 **SÍ, ES UNA COPIA / PLAGIO**")
                    st.write(f"Se ha detectado una copia no autorizada de: **{cancion_match}**.")
                    st.caption(f"Confianza del sistema: {mejor_similitud:.2f}%")
                
                else:
                    st.error("❌ **NO, NO ES UNA COPIA**")
                    st.write("No se encontraron coincidencias. El tema es original o una Cover IA con estructura acústica diferente.")

                # Bonus: Análisis visual
                with st.expander("Ver análisis técnico del espectrograma"):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    librosa.display.specshow(s_sospechoso, ax=ax, cmap='magma')
                    ax.set_title("Firma Espectral Capturada")
                    st.pyplot(fig)

except Exception as e:
    st.error(f"Error crítico del sistema: {e}")

st.divider()
st.caption("Proyecto de Tecnomatemática - Basado en redes neuronales convolucionales y autoencoders.")