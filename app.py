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

# --- ARQUITECTURA EXACTA DEL COLAB ---

class Autoencoder(nn.Module):
    def __init__(self, canales_latentes=8):
        super(Autoencoder, self).__init__()
        # Encoder: 1x256x256 -> 8x16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, canales_latentes, kernel_size=3, stride=2, padding=1), nn.Tanh()
        )
        # Decoder (necesario para que la clase sea idéntica al entrenamiento)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(canales_latentes, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh()
        )
    def forward(self, x):
        return self.encoder(x) # Solo necesitamos el encoder para el fingerprint

class EfficientNetFingerprint(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        c_orig = self.backbone.features[0][0]
        # Adaptación para los 8 canales que salen del autoencoder
        self.backbone.features[0][0] = nn.Conv2d(8, c_orig.out_channels, 3, stride=c_orig.stride, padding=c_orig.padding, bias=False)
        self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier[1].in_features, dim))
    def forward(self, x):
        # Redimensionamos el espacio latente para EfficientNet
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return F.normalize(self.backbone(x), p=2, dim=1)

# --- CARGA DE MODELOS ---
@st.cache_resource
def load_models():
    # Nota: Asegúrate que los archivos se llamen exactamente así en GitHub
    ae = Autoencoder(canales_latentes=8)
    ae.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'))
    ae.eval()
    
    cnn = EfficientNetFingerprint()
    cnn.load_state_dict(torch.load('cnn_fingerprint.pth', map_location='cpu'))
    cnn.eval()
    return ae, cnn

# --- PROCESAMIENTO ---
def get_fingerprint(audio_file, ae, cnn):
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # El Autoencoder espera 256x256
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        latente = ae.encoder(tensor_img) # Usamos solo la parte encoder
        fp = cnn(latente)
    return fp.numpy().flatten(), S_db

# --- INTERFAZ ---
st.set_page_config(page_title="Detector IA", page_icon="🎵")
st.title("🎵 Detector de Covers IA (Taller de Tecnomatematica)")

ae, cnn = load_models()

col1, col2 = st.columns(2)
with col1:
    orig = st.file_uploader("Audio Original", type=["mp3", "wav"])
with col2:
    susp = st.file_uploader("Audio Sospechoso", type=["mp3", "wav"])

if orig and susp:
    if st.button("ANALIZAR"):
        f1, s1 = get_fingerprint(orig, ae, cnn)
        f2, s2 = get_fingerprint(susp, ae, cnn)
        sim = np.dot(f1, f2) * 100
        
        st.metric("Similitud", f"{sim:.2f}%")
        if sim > 75:
            st.error("¡ALTA PROBABILIDAD DE COVER IA!")
        else:
            st.success("PARECE UNA CANCIÓN DIFERENTE")