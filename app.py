"""
Sistema de Monitoramento e Interpolação Climática - UFRRJ (Geociências)
Desenvolvido com Streamlit, GeoPandas, NumPy e Folium.
"""

import base64
import io
import logging
from typing import Tuple

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from folium import raster_layers
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from streamlit_folium import st_folium

# ==========================================
# CONFIGURAÇÕES E CONSTANTES
# ==========================================
st.set_page_config(page_title="Monitoramento Geociências", layout="wide")

# Delimitação do Prédio de Geociências (UFRRJ)
COORDENADAS_CAMPUS = [
    (-43.6882, -22.7688),
    (-43.6868, -22.7688),
    (-43.6868, -22.7700),
    (-43.6882, -22.7700)
]
POLIGONO_GEOCIENCIAS = Polygon(COORDENADAS_CAMPUS)

# Parâmetros de Renderização
MAPA_CENTRO = [-22.7694, -43.6875]
ZOOM_INICIAL = 18
RESOLUCAO_MALHA = 100

# ==========================================
# FUNÇÕES DE PROCESSAMENTO
# ==========================================
@st.cache_data(ttl=10)
def obter_dados_sensores() -> pd.DataFrame:
    """
    Estabelece conexão com o banco de dados e recupera as últimas leituras.
    """
    try:
        engine = create_engine(st.secrets["DB_URL"])
        query = """
            SELECT sensor_id, latitude, longitude, temperatura, data_hora 
            FROM medicoes 
            ORDER BY data_hora DESC 
            LIMIT 4
        """
        return pd.read_sql_query(query, engine)
    except SQLAlchemyError as e:
        logging.error(f"Falha na conexão com o banco de dados: {e}")
        st.error("Serviço temporariamente indisponível. Falha na conexão com o banco de dados.")
        return pd.DataFrame()

def estimar_temp_idw(lon_alvo: float, lat_alvo: float, df: pd.DataFrame, power: int = 2) -> float:
    """
    Algoritmo IDW: Estima a temperatura pontual usando o Inverso do Quadrado da Distância.
    """
    lons = df['longitude'].values
    lats = df['latitude'].values
    temps = df['temperatura'].values
    
    # Distância Euclidiana e prevenção de divisão por zero
    distancias = np.sqrt((lons - lon_alvo)**2 + (lats - lat_alvo)**2)
    distancias = np.where(distancias < 1e-10, 1e-10, distancias) 
    
    pesos = 1.0 / (distancias ** power)
    return float(np.sum(pesos * temps) / np.sum(pesos))

def gerar_camada_isolinhas(df: pd.DataFrame, limites: Tuple[float, float, float, float]) -> str:
    """
    Algoritmo IDW Vetorizado: Calcula a malha inteira e gera o mapa via Matplotlib.
    """
    min_lon, max_lon, min_lat, max_lat = limites

    grade_lon, grade_lat = np.meshgrid(
        np.linspace(min_lon, max_lon, RESOLUCAO_MALHA),
        np.linspace(min_lat, max_lat, RESOLUCAO_MALHA)
    )
    
    lons = df['longitude'].values
    lats = df['latitude'].values
    temps = df['temperatura'].values
    
    # Álgebra Linear para calcular a distância de 10.000 pontos instantaneamente
    distancias = np.sqrt((lons[:, None, None] - grade_lon)**2 + (lats[:, None, None] - grade_lat)**2)
    distancias = np.where(distancias < 1e-10, 1e-10, distancias)
    pesos = 1.0 / (distancias ** 2)
    
    temp_matriz = np.sum(pesos * temps[:, None, None], axis=0) / np.sum(pesos, axis=0)

    # Renderização científica
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off') 
    fig.patch.set_alpha(0.0) 
    ax.patch.set_alpha(0.0)

    ax.contourf(grade_lon, grade_lat, temp_matriz, levels=15, cmap='coolwarm', alpha=0.5)
    ax.contour(grade_lon, grade_lat, temp_matriz, levels=15, colors='black', linewidths=0.5, alpha=0.5)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"

# ==========================================
# INTERFACE PRINCIPAL (MAIN)
# ==========================================
def main():
    st.title("Mapa de Calor - Monitoramento Microclimático")
    st.markdown("---")

    df_sensores = obter_dados_sensores()
    
    if df_sensores.empty:
        st.stop()

    col_controle, col_mapa = st.columns([1, 3])

    with col_controle:
        st.subheader("Simulador de Posicionamento")
        st.markdown("Ajuste as coordenadas para estimar a temperatura local.")
        
        lon_usuario = st.slider("Longitude", -43.6890, -43.6860, -43.6875, format="%.5f")
        lat_usuario = st.slider("Latitude", -22.7710, -22.7680, -22.7694, format="%.5f")
        ponto_usuario = Point(lon_usuario, lat_usuario)

        tipo_mapa = st.radio(
            "Camada de Visualização:", 
            ["📍 Sensores e Posição Atual", "🌡️ Superfície Interpolada (Isolinhas)"]
        )

        st.markdown("---")
        
        if POLIGONO_GEOCIENCIAS.contains(ponto_usuario):
            # Chamada da nova função de IDW
            temp_estimada = estimar_temp_idw(lon_usuario, lat_usuario, df_sensores)
            
            st.success("✅ Coordenada válida (Área Interna)")
            st.metric(label="Temperatura Estimada", value=f"{temp_estimada:.2f} °C")
        else:
            temp_estimada = None
            st.error("❌ Fora da área de cobertura do projeto.")

    with col_mapa:
        mapa_base = folium.Map(location=MAPA_CENTRO, zoom_start=ZOOM_INICIAL)

        for _, linha in df_sensores.iterrows():
            folium.CircleMarker(
                location=[linha['latitude'], linha['longitude']],
                radius=6, color="#333333", fill=True, fill_color="#ff4444", fill_opacity=1,
                tooltip=f"Sensor {linha['sensor_id']} | {linha['temperatura']}°C"
            ).add_to(mapa_base)

        if tipo_mapa == "🌡️ Superfície Interpolada (Isolinhas)":
            min_lon, max_lon = -43.6882, -43.6868
            min_lat, max_lat = -22.7700, -22.7688
            limites_terreno = (min_lon, max_lon, min_lat, max_lat)

            # Nova rotina vetorizada que nunca falha matematicamente
            camada_imagem = gerar_camada_isolinhas(df_sensores, limites_terreno)

            raster_layers.ImageOverlay(
                image=camada_imagem,
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                opacity=0.6,
                interactive=False,
                cross_origin=False,
                zindex=1
            ).add_to(mapa_base)

        if temp_estimada is not None:
            folium.Marker(
                location=[lat_usuario, lon_usuario],
                popup=folium.Popup(f"<b>Temperatura:</b> {temp_estimada:.1f}°C", max_width=200),
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(mapa_base)

        st_folium(
            mapa_base, 
            height=500, 
            use_container_width=True, 
            returned_objects=[], 
            key=f"render_{tipo_mapa}"
        )

if __name__ == "__main__":
    main()
