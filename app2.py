import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Music Fingerprint Debugger", layout="wide")

# --- DEFINIZIONE MODELLI (Assicurati che siano uguali ai tuoi) ---
class Autoencoder(nn.Module):
    def __init__(self, canales_latentes=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class EfficientNetFingerprint(nn.Module):
    def __init__(self, fingerprint_size=128):
        super(EfficientNetFingerprint, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, fingerprint_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Normalizzazione L2 per rendere il prodotto punto uguale alla similitudine coseno
        return x / torch.norm(x, p=2, dim=1, keepdim=True)

# --- FUNZIONI DI CARICAMENTO ---
@st.cache_resource
def load_assets():
    ae = Autoencoder(8)
    cnn = EfficientNetFingerprint(128)
    # Carica i pesi (assicurati che i file siano su GitHub)
    ae.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'), strict=False)
    cnn.load_state_dict(torch.load('cnn_fingerprint.pth', map_location='cpu'), strict=False)
    ae.eval()
    cnn.eval()
    db = torch.load('database_impronte.pt', map_location='cpu')
    return ae, cnn, db

# --- FUNZIONE DI ELABORAZIONE ---
def process_input(file_bytes, is_image, ae, cnn):
    if is_image:
        # Se è un'immagine, la carichiamo direttamente
        img_pil = Image.open(io.BytesIO(file_bytes)).convert('L')
    else:
        # Se è audio, generiamo lo spettrogramma
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
        D = np.abs(librosa.stft(y))**2 
        S = librosa.power_to_db(D, ref=np.max)
        
        fig = plt.figure(figsize=(4, 4))
        librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log")
        plt.axis("off")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img_pil = Image.open(buf).convert('L')

    # Resize e Trasformazione (Uguale al test di Colab)
    img_resized = img_pil.resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_img = transform(img_resized).unsqueeze(0)

    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    
    return fp.numpy().flatten(), img_pil

# --- INTERFACCIA STREAMLIT ---
st.title("🎵 Music Recognition Debugger")
ae, cnn, db_huellas = load_assets()

# Accettiamo sia MP3/WAV che PNG
uploaded_file = st.file_uploader("Carica un file Audio o lo Spettrogramma PNG", type=["mp3", "wav", "png"])

if uploaded_file is not None:
    is_png = uploaded_file.name.lower().endswith('.png')
    file_bytes = uploaded_file.read()
    
    huella_input, img_visual = process_input(file_bytes, is_png, ae, cnn)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_visual, caption="Spettrogramma processato", use_container_width=True)
    
    # Ricerca nel Database
    max_sim = -1
    best_name = ""
    
    for name, huella_db in db_huellas.items():
        sim = np.dot(huella_input, huella_db)
        if sim > max_sim:
            max_sim = sim
            best_name = name
            
    with col2:
        st.subheader("Risultato Analisi")
        st.write(f"**Miglior Match:** {best_name}")
        st.write(f"**Similitudine:** {max_sim:.4f}")
        
        if max_sim > 0.95:
            st.success("✅ Canzone Identificata con successo!")
        elif max_sim > 0.70:
            st.warning("⚠️ Somiglianza alta: Possibile Cover o versione alternativa.")
        else:
            st.error("❌ Nessuna corrispondenza trovata.")

    # DEBUG SECTION
    with st.expander("Vedi Debug Database"):
        st.write(f"Canzoni nel DB: {len(db_huellas)}")
        st.write("Esempio nomi nel DB:", list(db_huellas.keys())[:5])