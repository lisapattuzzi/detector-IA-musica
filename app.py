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

# --- ARQUITECTURA (Debe ser idéntica a tu Colab) ---
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
        x = F.interpolate(x, size=(64, 64), mode='bilinear')
        return F.normalize(self.backbone(x), p=2, dim=1)

# --- FUNCIONES DE APOYO ---
@st.cache_resource
def load_models():
    ae = Autoencoder()
    ae.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'))
    ae.eval()
    cnn = EfficientNetFingerprint()
    cnn.load_state_dict(torch.load('cnn_fingerprint.pth', map_location='cpu'))
    cnn.eval()
    return ae, cnn

def get_fingerprint(audio_file, ae, cnn):
    # Carga de audio y generación de espectrograma
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Preprocesamiento para los modelos
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    tensor_img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        latent = ae(tensor_img)
        fp = cnn(latent)
    return fp.numpy().flatten(), S_db

# --- INTERFAZ DE USUARIO (STREAMLIT) ---
st.set_page_config(page_title="Detector de Covers IA", page_icon="🎵")

st.markdown("""
    <style>
    .title-text { font-size: 40px; font-weight: 800; color: #1E3A8A; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title-text">🎵 Detector de Covers IA y Plagio</p>', unsafe_allow_html=True)
st.write("---")
st.info("Cargue la canción original y el archivo sospechoso para comparar sus huellas digitales (fingerprints) matemáticas.")

ae, cnn = load_models()

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Canción Original")
    orig = st.file_uploader("Subir archivo original", type=["mp3", "wav"], key="orig")
with col2:
    st.subheader("2. Canción Sospechosa")
    susp = st.file_uploader("Subir archivo sospechoso", type=["mp3", "wav"], key="susp")

if orig and susp:
    if st.button("🔍 INICIAR ANÁLISIS MATEMÁTICO"):
        with st.spinner("Procesando señales acústicas..."):
            fp1, spec1 = get_fingerprint(orig, ae, cnn)
            fp2, spec2 = get_fingerprint(susp, ae, cnn)
            
            # Cálculo de similitud del coseno
            sim = np.dot(fp1, fp2) * 100
            
            st.divider()
            st.subheader("Resultado del Análisis")
            
            # Métrica visual
            st.metric("Similitud Encontrada", f"{sim:.2f}%")
            
            if sim > 80:
                st.error("🚨 RESULTADO: COVER IA / PLAGIO DETECTADO")
                st.write("**Conclusión:** Las estructuras matemáticas de ambas canciones coinciden significativamente en el espacio latente.")
            elif sim > 50:
                st.warning("⚠️ RESULTADO: POSIBLE DERIVACIÓN")
                st.write("**Conclusión:** Se detectan similitudes estructurales, pero con variaciones notables.")
            else:
                st.success("✅ RESULTADO: CANCIÓN ORIGINAL / DIFERENTE")
                st.write("**Conclusión:** No se encontraron vínculos matemáticos. Las huellas digitales son distintas.")

            # Mostrar Espectrogramas para el Prof.
            with st.expander("Ver Análisis Espectral (Espectrogramas de Mel)"):
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                librosa.display.specshow(spec1, x_axis='time', y_axis='mel', ax=ax[0])
                ax[0].set_title("Original")
                librosa.display.specshow(spec2, x_axis='time', y_axis='mel', ax=ax[1])
                ax[1].set_title("Sospechosa")
                st.pyplot(fig)