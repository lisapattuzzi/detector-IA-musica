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

# --- ARCHITETTURE ---
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

def load_weights_safe(model, path):
    state_dict = torch.load(path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
    # strict=False ignora le chiavi del decoder che non abbiamo definito qui
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

@st.cache_resource
def load_models():
    ae = Autoencoder(canales_latentes=8)
    ae = load_weights_safe(ae, 'autoencoder.pth')
    cnn = EfficientNetFingerprint()
    cnn = load_weights_safe(cnn, 'cnn_fingerprint.pth')
    return ae, cnn

def get_fingerprint(audio_file, ae, cnn):
    y, sr = librosa.load(audio_file, duration=10)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    tensor_img = transform(img).unsqueeze(0)
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    return fp.numpy().flatten(), S_db

# --- INTERFACCIA ---
st.set_page_config(page_title="Detector de Covers IA", page_icon="🎵", layout="wide")
st.title("🎵 Detector de Covers IA y Plagio")
st.markdown("### Taller de Tecnomatematica")

try:
    ae, cnn = load_models()
    col1, col2 = st.columns(2)
    with col1:
        u1 = st.file_uploader("Subir Canción Original", type=["mp3", "wav"])
    with col2:
        u2 = st.file_uploader("Subir Canción Sospechosa", type=["mp3", "wav"])

    if u1 and u2:
        if st.button("🚀 CALCULAR SIMILITUD"):
            with st.spinner("Analizando huellas digitales..."):
                f1, s1 = get_fingerprint(u1, ae, cnn)
                f2, s2 = get_fingerprint(u2, ae, cnn)
                sim = np.dot(f1, f2) * 100
                st.divider()
                st.header(f"Similitud: {sim:.2f}%")
                if sim > 75:
                    st.error("🚨 RESULTADO: COVER IA / PLAGIO DETECTADO")
                else:
                    st.success("✅ RESULTADO: CANCIÓN ORIGINAL / DIFERENTE")
                
                fig, ax = plt.subplots(1, 2, figsize=(10, 3))
                librosa.display.specshow(s1, ax=ax[0], cmap='magma')
                ax[0].set_title("Original")
                librosa.display.specshow(s2, ax=ax[1], cmap='magma')
                ax[1].set_title("Sospechosa")
                st.pyplot(fig)
    else:
        st.info("👋 Sube ambos audios para comenzar.")
except Exception as e:
    st.error(f"Error crítico: {e}")