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

# --- ARQUITECTURA DEL AUTOENCODER (IDÉNTICA A COLAB) ---
class Autoencoder(nn.Module):
    def __init__(self, canales_latentes=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, canales_latentes, kernel_size=3, stride=2, padding=1), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(canales_latentes, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh()
        )
    def forward(self, x): return self.encoder(x)

# --- ARQUITECTURA DE LA CNN (ADAPTADA A 8 CANALES) ---
class EfficientNetFingerprint(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # Importante: weights=None porque los cargaremos del archivo .pth
        self.backbone = models.efficientnet_b0(weights=None)
        c_orig = self.backbone.features[0][0]
        # Ajuste crítico: la entrada debe ser de 8 canales (lo que sale del Autoencoder)
        self.backbone.features[0][0] = nn.Conv2d(8, c_orig.out_channels, 3, stride=c_orig.stride, padding=c_orig.padding, bias=False)
        self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier[1].in_features, dim))
    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return F.normalize(self.backbone(x), p=2, dim=1)

# --- CARGA DE MODELOS CON MANEJO DE ERRORES ---
@st.cache_resource
def load_models():
    ae = Autoencoder(canales_latentes=8)
    # Asegúrate que el archivo se llame exactamente autoencoder.pth
    ae.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'))
    ae.eval()
    
    cnn = EfficientNetFingerprint()
    # Asegúrate que el archivo se llame exactamente cnn_fingerprint.pth
    cnn.load_state_dict(torch.load('cnn_fingerprint.pth', map_location='cpu'))
    cnn.eval()
    return ae, cnn

# --- FUNCIÓN DE FINGERPRINTING ---
def get_fingerprint(audio_file, ae, cnn):
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Preprocesamiento
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
st.title("🎵 Detector IA: Análisis de Huellas Digitales")
ae, cnn = load_models()

u1 = st.file_uploader("Canción A (Original)", type=["mp3", "wav"])
u2 = st.file_uploader("Canción B (Sospechosa)", type=["mp3", "wav"])

if u1 and u2:
    if st.button("CALCULAR SIMILITUD"):
        f1 = get_fingerprint(u1, ae, cnn)
        f2 = get_fingerprint(u2, ae, cnn)
        sim = np.dot(f1, f2) * 100
        st.metric("Nivel de Coincidencia", f"{sim:.2f}%")