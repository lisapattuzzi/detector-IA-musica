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
from collections import OrderedDict # Corregido: OrderedDict

# --- ARQUITECTURAS ---

class Autoencoder(nn.Module):
    def __init__(self, canales_latentes=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, canales_latentes, 3, stride=2, padding=1), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(canales_latentes, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Tanh()
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

# --- FUNCIÓN PARA LIMPIAR PESOS ---
def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # Elimina el prefijo si se entrenó con DataParallel
        new_state_dict[name] = v
    return new_state_dict

@st.cache_resource
def load_models():
    # Cargar Autoencoder
    ae = Autoencoder(canales_latentes=8)
    state_ae = torch.load('autoencoder.pth', map_location='cpu')
    ae.load_state_dict(clean_state_dict(state_ae))
    ae.eval()
    
    # Cargar CNN
    cnn = EfficientNetFingerprint()
    state_cnn = torch.load('cnn_fingerprint.pth', map_location='cpu')
    cnn.load_state_dict(clean_state_dict(state_cnn))
    cnn.eval()
    return ae, cnn

# --- PROCESAMIENTO ---
def get_fingerprint(audio_file, ae, cnn):
    # Cargamos 10 segundos para consistencia
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    return fp.numpy().flatten()

# --- INTERFAZ ---
st.set_page_config(page_title="Detector de Covers IA", page_icon="🎵")
st.title("🎵 Detector de Covers IA y Plagio")
st.markdown("### Taller de Tecnomatematica")

ae, cnn = load_models()

u1 = st.file_uploader("Subir Canción Original", type=["mp3", "wav"])
u2 = st.file_uploader("Subir Canción Sospechosa", type=["mp3", "wav"])

if u1 and u2:
    if st.button("CALCULAR SIMILITUD"):
        with st.spinner("Analizando huellas digitales musicales..."):
            f1 = get_fingerprint(u1, ae, cnn)
            f2 = get_fingerprint(u2, ae, cnn)
            
            # Cálculo de Similitud Coseno
            sim = np.dot(f1, f2) * 100
            
            st.divider()
            st.metric("Nivel de Similitud", f"{sim:.2f}%")
            
            if sim > 75:
                st.error("🚨 RESULTADO: COVER IA / PLAGIO DETECTADO")
                st.write("La estructura matemática es extremadamente similar.")
            else:
                st.success("✅ RESULTADO: BRANO ORIGINALE / DIVERSO")
                st.write("Las huellas digitales no coinciden significativamente.")