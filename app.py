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

# ==========================================================
# 1. DEFINICIÓN DE ARQUITECTURAS
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
        self.backbone.features[0][0] = nn.Conv2d(8, c_orig.out_channels, 3, stride=c_orig.stride, padding=c_orig.padding, bias=False)
        self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier[1].in_features, dim))
    
    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return F.normalize(self.backbone(x), p=2, dim=1)

# ==========================================================
# 2. CARGA DE MODELOS (Caché para mayor velocidad)
# ==========================================================

@st.cache_resource
def load_assets():
    ae = Autoencoder(canales_latentes=8)
    cnn = EfficientNetFingerprint()
    db = None
    
    if os.path.exists('autoencoder.pth'):
        state_dict = torch.load('autoencoder.pth', map_location='cpu')
        ae.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
        ae.eval()
        
    if os.path.exists('cnn_fingerprint.pth'):
        state_dict = torch.load('cnn_fingerprint.pth', map_location='cpu')
        cnn.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
        cnn.eval()
        
    if os.path.exists('database_impronte.pt'):
        db = torch.load('database_impronte.pt', map_location='cpu', weights_only=False)
        
    return ae, cnn, db

# ==========================================================
# 3. PROCESAMIENTO AUTOMÁTICO EXACTO (Matriz a Imagen directa)
# ==========================================================

def process_audio_chunk(y_chunk, sr, ae, cnn):
    """Genera una imagen limpia, cuadrada y magma directamente de la matriz"""
    # 1. Generar Espectrograma Mel (128 mels)
    S = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max) # Shape (128, X)

    # 2. Normalizar la matriz a valores de 0 a 255
    S_norm = (S_db + 80) / 80 * 255
    S_norm = np.clip(S_norm, 0, 255).astype(np.uint8)

    # 3. Voltear verticalmente para coincidir con la orientación de un gráfico normal
    S_flipped = np.flip(S_norm, axis=0)

    # 4. Redimensionar a (256, 256) usando PIL para tamaño exacto sin bordes
    img_gray = Image.fromarray(S_flipped, mode='L')
    img_resized = img_gray.resize((256, 256), Image.Resampling.BILINEAR)

    # 5. Aplicar mapa de colores Magma a los datos crudos
    gray_arr = np.array(img_resized)
    cmap = plt.get_cmap('magma')
    colored_arr = (cmap(gray_arr / 255.0) * 255).astype(np.uint8)

    # 6. Convertir a imagen RGB final para la red neuronal
    img_pil = Image.fromarray(colored_arr, mode='RGBA').convert('RGB')
    
    # 7. Transformar imagen para la red (Normalización a escala de grises para Autoencoder)
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img_pil).unsqueeze(0)
    
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
        
    return fp.numpy().flatten(), S_db

# ==========================================================
# 4. INTERFAZ DE USUARIO LIMPIA
# ==========================================================

st.set_page_config(page_title="Copyright Detector", page_icon="🎵", layout="centered")

st.title("🎵 Escáner de Plagio y Copyright")
st.markdown("Sube una canción y el sistema la analizará automáticamente contra la base de datos mediante Ventana Deslizante.")

# Cargar sistema
ae, cnn, db_huellas = load_assets()

if db_huellas is None:
    st.error("⚠️ Base de datos no encontrada. Asegúrate de tener 'database_impronte.pt' en tu repositorio.")
    st.stop()

# Interfaz de carga
audio_file = st.file_uploader("📁 Selecciona un archivo MP3 o WAV", type=["mp3", "wav"])

if audio_file:
    if st.button("🚀 INICIAR ANÁLISIS COMPLETO", use_container_width=True):
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # 1. Forzamos la carga a 22050Hz (el estándar con el que seguro se entrenó la red)
        status_text.info("Cargando y normalizando audio (22050 Hz)...")
        try:
            y_full, sr = librosa.load(audio_file, sr=22050)
        except Exception as e:
            st.error(f"Error al cargar el audio: {e}")
            st.stop()
        
        # 2. Configurar ventana deslizante: trozos de 10 seg, avanzando de 5 en 5 seg
        chunk_samples = 10 * sr
        step_samples = 5 * sr
        
        if len(y_full) < chunk_samples:
            y_full = np.pad(y_full, (0, chunk_samples - len(y_full)))
            
        num_chunks = max(1, (len(y_full) - chunk_samples) // step_samples + 1)
        
        mejor_sim_global = -100
        mejor_match_global = ""
        top_3_global = []
        espectrograma_visual = None

        # 3. Escanear toda la canción
        for i in range(num_chunks):
            status_text.warning(f"🔍 Escaneando fragmento {i+1} de {num_chunks}...")
            
            start = i * step_samples
            end = start + chunk_samples
            y_chunk = y_full[start:end]
            
            # Obtener vector del fragmento
            fp_chunk, s_chunk_db = process_audio_chunk(y_chunk, sr, ae, cnn)
            
            # Comparar con DB
            resultados_segmento = []
            for nombre, fp_ref in db_huellas.items():
                sim = np.dot(fp_chunk, fp_ref) * 100
                resultados_segmento.append((nombre, sim))
            
            resultados_segmento.sort(key=lambda x: x[1], reverse=True)
            mejor_match_chunk, mejor_sim_chunk = resultados_segmento[0]
            
            # Guardar el mejor resultado histórico de la canción
            if mejor_sim_chunk > mejor_sim_global:
                mejor_sim_global = mejor_sim_chunk
                mejor_match_global = mejor_match_chunk
                top_3_global = resultados_segmento[:3]
                espectrograma_visual = s_chunk_db
            
            progress_bar.progress((i + 1) / num_chunks)
            
            # Optimización: Si encontramos una coincidencia casi perfecta, paramos para ahorrar tiempo
            if mejor_sim_global > 95:
                progress_bar.progress(1.0)
                break
        
        status_text.empty() # Limpiar mensajes de carga

        # 4. Mostrar Resultados Finales
        st.divider()
        st.header("Veredicto Final")

        if mejor_sim_global >= 90:
            st.success(f"✅ **COINCIDENCIA EXACTA CONFIRMADA**\n\nIdentificada como: **{mejor_match_global}**")
        elif mejor_sim_global >= 60:
            st.warning(f"🚨 **ALERTA DE POSIBLE PLAGIO**\n\nAlta correlación con: **{mejor_match_global}**")
        else:
            st.error("❌ **AUDIO LIMPIO**\n\nNo se encontraron copias en la base de datos.")

        # Sección para el profesor (Visualización estética)
        with st.expander("📊 Ver Detalles del Análisis (Top 3)"):
            st.write(f"Porcentaje de confianza máximo encontrado en el audio: **{mejor_sim_global:.2f}%**")
            for i, (nombre, sim) in enumerate(top_3_global):
                st.write(f"**{i+1}.** {nombre} ➔ `{sim:.2f}%`")
            
            st.markdown("### 🎛️ Huella Acústica Detectada")
            # Gráfico seguro para mostrar al profesor (Esta imagen NO va a la red neuronal)
            fig_debug, ax_debug = plt.subplots(figsize=(8, 3))
            img = librosa.display.specshow(espectrograma_visual, ax=ax_debug, sr=sr, cmap='magma', y_axis='mel', x_axis='time')
            plt.colorbar(img, format='%+2.0f dB', ax=ax_debug)
            st.pyplot(fig_debug)