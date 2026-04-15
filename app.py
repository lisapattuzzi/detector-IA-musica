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

# --- ARCHITETTURE (Le stesse del tuo modello funzionante) ---
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
    try:
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Impossibile caricare il file {path}.")

@st.cache_resource
def load_models():
    ae = Autoencoder(canales_latentes=8)
    ae = load_weights_safe(ae, 'autoencoder.pth')
    cnn = EfficientNetFingerprint()
    cnn = load_weights_safe(cnn, 'cnn_fingerprint.pth')
    return ae, cnn

def get_fingerprint(audio_file, ae, cnn):
    y, sr = librosa.load(audio_file, duration=10) # Analizziamo i primi 10 secondi
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = Image.fromarray(((S_db + 80) / 80 * 255).astype(np.uint8)).resize((256, 256))
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    tensor_img = transform(img).unsqueeze(0)
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
    return fp.numpy().flatten(), S_db

# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="Music Plagiarism Scanner", page_icon="🔍", layout="wide")
st.title("🔍 Scanner di Plagio e Cover Musicali")
st.markdown("### Taller de Tecnomatematica - Ricerca nel Dataset")

try:
    ae, cnn = load_models()
    
    st.markdown("### 1. Costruisci il tuo Dataset")
    st.write("Carica le canzoni originali che vuoi inserire nel tuo database di riferimento.")
    dataset_files = st.file_uploader("Carica una o più canzoni per il Dataset", type=["mp3", "wav"], accept_multiple_files=True)

    st.markdown("### 2. Carica la traccia Sospetta")
    st.write("Carica la canzone in ingresso per verificare se è un plagio o una cover di una delle canzoni nel dataset.")
    suspect_file = st.file_uploader("Carica il brano da analizzare", type=["mp3", "wav"])

    if dataset_files and suspect_file:
        if st.button("🚀 SCANSIONA IL DATASET"):
            with st.spinner("Estrazione impronte digitali e ricerca in corso..."):
                
                # 1. Calcoliamo l'impronta del brano sospetto
                f_sospetto, s_sospetto = get_fingerprint(suspect_file, ae, cnn)
                
                miglior_match = None
                similitudine_massima = -1
                spettrogramma_migliore = None
                
                # 2. Cicliamo su tutte le canzoni del dataset per trovare quella più simile
                for original_file in dataset_files:
                    f_originale, s_originale = get_fingerprint(original_file, ae, cnn)
                    
                    # Prodotto scalare (Similitudine)
                    sim = np.dot(f_sospetto, f_originale) * 100
                    
                    if sim > similitudine_massima:
                        similitudine_massima = sim
                        miglior_match = original_file.name
                        spettrogramma_migliore = s_originale
                
                # 3. Mostriamo i risultati
                st.divider()
                st.header(f"Risultato della Ricerca")
                st.subheader(f"La canzone più simile nel dataset è: **{miglior_match}**")
                st.metric("Indice di Similitudine", f"{similitudine_massima:.2f}%")
                
                # --- LOGICA DI CLASSIFICAZIONE ---
                if similitudine_massima > 85:
                    st.error("🚨 RISULTATO: PLAGIO DIRETTO / IDENTITÀ ACUSTICA")
                    st.write("Il brano in ingresso è matematicamente identico (o quasi) alla canzone trovata nel dataset.")
                elif similitudine_massima > 20:
                    st.warning("⚠️ RISULTATO: POSSIBILE CAMPIONAMENTO / REMIX")
                    st.write("Il brano ha elementi in comune con la canzone del dataset, ma presenta modifiche significative.")
                else:
                    st.info("💡 RISULTATO: COVER IA / NESSUN PLAGIO DIRETTO")
                    st.write("La similitudine è molto bassa. Potrebbe trattarsi di un brano totalmente diverso, o di una Cover IA che ha alterato completamente la struttura dello spettrogramma.")

                # Visualizzazione
                st.markdown("### Confronto Spettrogrammi con il miglior Match")
                fig, ax = plt.subplots(1, 2, figsize=(10, 3))
                librosa.display.specshow(spettrogramma_migliore, ax=ax[0], cmap='magma')
                ax[0].set_title(f"Dataset: {miglior_match}")
                librosa.display.specshow(s_sospetto, ax=ax[1], cmap='magma')
                ax[1].set_title("Brano Sospetto")
                st.pyplot(fig)
    else:
        st.info("👋 Carica almeno una canzone nel Dataset e una canzone Sospetta per iniziare.")

except Exception as e:
    st.error(f"Errore tecnico: {e}")