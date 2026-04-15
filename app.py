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
# 2. CARGA DE MODELOS
# ==========================================================

def load_weights_safe(model, path):
    if not os.path.exists(path):
        st.error(f"⚠️ Archivo de pesos no encontrado: {path}")
        return model
    try:
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error cargando {path}: {e}")
        return model

@st.cache_resource
def load_assets():
    ae = Autoencoder(canales_latentes=8)
    ae = load_weights_safe(ae, 'autoencoder.pth')
    cnn = EfficientNetFingerprint()
    cnn = load_weights_safe(cnn, 'cnn_fingerprint.pth')
    
    db_path = 'database_impronte.pt'
    db = None
    if os.path.exists(db_path):
        db = torch.load(db_path, map_location='cpu', weights_only=False)
    return ae, cnn, db

# ==========================================================
# 3. MOTOR DE PROCESAMIENTO POR VENTANA DESLIZANTE
# ==========================================================

def process_audio_chunk(y_chunk, sr, ae, cnn):
    """Procesa un fragmento de 10 segundos y devuelve su vector"""
    S = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Recrear el formato visual exacto del dataset original
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.axis('off')
    librosa.display.specshow(S_db, sr=sr, ax=ax, cmap='magma')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert('L').resize((256, 256))
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    tensor_img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        latent = ae.encoder(tensor_img)
        fp = cnn(latent)
        
    return fp.numpy().flatten(), S_db

# ==========================================================
# 4. INTERFAZ STREAMLIT
# ==========================================================

st.set_page_config(page_title="Copyright Detector", page_icon="🎵")
st.title("🎵 Detector de Copyright y Plagio")
st.markdown("### Escáner Profundo (Análisis de Ventana Deslizante)")

try:
    ae, cnn, db_huellas = load_assets()

    if db_huellas is None:
        st.error("⚠️ Falta la base de datos 'database_impronte.pt'.")
        st.stop()

    with st.expander("📚 Ver base de datos cargada"):
        st.write(list(db_huellas.keys()))

    st.divider()
    
    tab1, tab2 = st.tabs(["📁 Cargar Archivo", "🎙️ Micrófono"])
    with tab1:
        audio_file = st.file_uploader("Sube el archivo de audio completo", type=["mp3", "wav"])
    with tab2:
        audio_input = st.audio_input("Graba audio")

    u_audio = audio_file if audio_file else audio_input

    if u_audio:
        if st.button("🚀 INICIAR ESCÁNER PROFUNDO", use_container_width=True):
            
            # Cargar todo el audio en la memoria
            y_full, sr = librosa.load(u_audio, sr=None)
            
            # Configuramos la "Ventana Deslizante": Trozos de 10 seg, avanzando cada 5 seg
            chunk_samples = 10 * sr
            step_samples = 5 * sr
            
            # Si el audio es muy corto, lo rellenamos
            if len(y_full) < chunk_samples:
                y_full = np.pad(y_full, (0, chunk_samples - len(y_full)))
                
            num_chunks = max(1, (len(y_full) - chunk_samples) // step_samples + 1)
            
            st.info(f"El audio se dividió en **{num_chunks} segmentos** para buscar coincidencias ocultas.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Variables para guardar el mejor resultado de toda la canción
            mejor_sim_global = -100
            mejor_match_global = ""
            mejor_espectrograma = None
            top_3_global = []

            # ==========================================
            # BUCLE DE ESCANEO (Ventana Deslizante)
            # ==========================================
            for i in range(num_chunks):
                status_text.text(f"Analizando segmento {i+1} de {num_chunks}...")
                
                start = i * step_samples
                end = start + chunk_samples
                y_chunk = y_full[start:end]
                
                fp_chunk, s_chunk = process_audio_chunk(y_chunk, sr, ae, cnn)
                
                # Comparar este segmento específico con toda la base de datos
                resultados_segmento = []
                for nombre, fp_ref in db_huellas.items():
                    sim = np.dot(fp_chunk, fp_ref) * 100
                    resultados_segmento.append((nombre, sim))
                
                # Ordenar los resultados de este trocito
                resultados_segmento.sort(key=lambda x: x[1], reverse=True)
                mejor_match_chunk, mejor_sim_chunk = resultados_segmento[0]
                
                # Si este trocito tiene una coincidencia más alta que las anteriores, lo guardamos
                if mejor_sim_chunk > mejor_sim_global:
                    mejor_sim_global = mejor_sim_chunk
                    mejor_match_global = mejor_match_chunk
                    mejor_espectrograma = s_chunk
                    top_3_global = resultados_segmento[:3]
                
                # Actualizar barra de progreso
                progress_bar.progress((i + 1) / num_chunks)
            
            status_text.success("¡Análisis completado!")

            # ==========================================
            # VEREDICTO FINAL
            # ==========================================
            st.divider()
            st.header("Resultado del Análisis")

            if mejor_sim_global >= 90:
                st.success(f"✅ **ES LA CANCIÓN ORIGINAL**\n\nCoincidencia exacta con: **{mejor_match_global}**")
            elif mejor_sim_global >= 60:
                st.warning(f"🚨 **ALERTA DE PLAGIO / MODIFICACIÓN**\n\nAlta correlación con: **{mejor_match_global}**")
            else:
                st.error("❌ **NO ES UNA COPIA**\n\nNo se encontraron coincidencias suficientes.")

            # DETALLES TÉCNICOS
            with st.expander("⚙️ Ver Detalles Matemáticos (Del mejor segmento encontrado)"):
                st.write(f"La similitud más alta se encontró con una confianza del **{mejor_sim_global:.2f}%**.")
                st.markdown("### Top 3 Similitudes:")
                for i, (nombre, sim) in enumerate(top_3_global):
                    st.write(f"**{i+1}.** {nombre} ➔ `{sim:.2f}%`")
                
                st.divider()
                st.markdown("### 🎛️ Espectrograma del segmento coincidente")
                fig_debug, ax_debug = plt.subplots(figsize=(8, 3))
                librosa.display.specshow(mejor_espectrograma, ax=ax_debug, cmap='magma', y_axis='mel', x_axis='time')
                st.pyplot(fig_debug)

except Exception as e:
    st.error(f"Error crítico: {e}")