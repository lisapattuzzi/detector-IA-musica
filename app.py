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

# [Le classi Autoencoder e EfficientNetFingerprint rimangono identiche a prima]
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

@st.cache_resource
def load_assets():
    ae, cnn = Autoencoder(8), EfficientNetFingerprint()
    if os.path.exists('autoencoder.pth'): ae.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'), strict=False)
    if os.path.exists('cnn_fingerprint.pth'): cnn.load_state_dict(torch.load('cnn_fingerprint.pth', map_location='cpu'), strict=False)
    db = torch.load('database_impronte.pt', map_location='cpu') if os.path.exists('database_impronte.pt') else None
    ae.eval(); cnn.eval()
    return ae, cnn, db

def process_audio_final(audio_file, ae, cnn):
    # 1. CARICAMENTO: Forziamo il Sample Rate a 22050 o 44100? 
    # Proviamo con sr=None come nel tuo codice originale
    y, sr = librosa.load(audio_file, sr=None)
    
    # 2. STFT (Esattamente come il tuo snippet)
    D = np.abs(librosa.stft(y))**2
    S = librosa.power_to_db(D, ref=np.max)
    
    # 3. GENERAZIONE IMMAGINE (Forziamo i DPI per evitare variazioni di dimensione)
    fig = plt.figure(figsize=(4, 4), dpi=100)
    librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log")
    plt.axis("off")
    
    buf = io.BytesIO()
    # Usiamo lo stesso identico metodo di salvataggio del dataset
    plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    # 4. PREPARAZIONE PER LA RETE
    img = Image.open(buf).convert('L').resize((256, 256))
    # Importante: Non usare Grayscale() qui perché è già 'L', carichiamo il tensor diretto
    tensor_img = transforms.ToTensor()(img).unsqueeze(0)
    tensor_img = (tensor_img - 0.5) / 0.5 # Normalizzazione manuale equivalente
    
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
        
    return fp.numpy().flatten(), img

# --- INTERFACCIA ---
st.title("🎵 Detector de Copyright (STFT Exacto)")
ae, cnn, db = load_assets()
file = st.file_uploader("Sube audio", type=["mp3", "wav"])

if file and st.button("Analizar"):
    fp, img_vis = process_audio_final(file, ae, cnn)
    res = sorted([(n, np.dot(fp, f_ref)*100) for n, f_ref in db.items()], key=lambda x: x[1], reverse=True)
    
    st.image(img_vis, caption="Spettrogramma Generato")
    st.write(f"### Resultado: {res[0][0]} ({res[0][1]:.2f}%)")
    with st.expander("Top 5"):
        for n, s in res[:5]: st.write(f"{n}: {s:.2f}%")