"""
Sistema de Monitoramento e Interpolação Climática - UFRRJ (Geociências)
Desenvolvido com Streamlit, GeoPandas, NumPy e Folium.
"""

import base64
import io
import logging
from typing import Tuple
import os
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
st.set_page_config(page_title="Monitoramento Geociências", page_icon="🌍", layout="centered")

# Delimitação do Prédio de Geociências
COORDENADAS_CAMPUS = [
    (-43.6882, -22.7688),
    (-43.6868, -22.7688),
    (-43.6868, -22.7700),
    (-43.6882, -22.7700)
]
POLIGONO_GEOCIENCIAS = Polygon(COORDENADAS_CAMPUS)

# Parâmetros de Renderização do Mapa
MAPA_CENTRO = [-22.7694, -43.6875]
ZOOM_INICIAL = 18
RESOLUCAO_MALHA = 100

# ==========================================
# FUNÇÕES DE PROCESSAMENTO
# ==========================================
@st.cache_data(ttl=10)
def obter_dados_sensores() -> pd.DataFrame:
    """Recupera as leituras mais recentes do banco de dados (Supabase)."""
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
        logging.error(f"Falha na conexão: {e}")
        st.error("Serviço temporariamente indisponível. Falha na conexão com o banco de dados.")
        return pd.DataFrame()

def estimar_temp_idw(lon_alvo: float, lat_alvo: float, df: pd.DataFrame, power: int = 2) -> float:
    """Calcula a temperatura exata de um ponto usando o Inverso do Quadrado da Distância."""
    lons = df['longitude'].values
    lats = df['latitude'].values
    temps = df['temperatura'].values
    
    distancias = np.sqrt((lons - lon_alvo)**2 + (lats - lat_alvo)**2)
    distancias = np.where(distancias < 1e-10, 1e-10, distancias) 
    
    pesos = 1.0 / (distancias ** power)
    return float(np.sum(pesos * temps) / np.sum(pesos))

def gerar_camada_isolinhas(df: pd.DataFrame, limites: Tuple[float, float, float, float]) -> str:
    """Gera o mapa de calor científico (isopletas) e o converte para imagem."""
    min_lon, max_lon, min_lat, max_lat = limites

    grade_lon, grade_lat = np.meshgrid(
        np.linspace(min_lon, max_lon, RESOLUCAO_MALHA),
        np.linspace(min_lat, max_lat, RESOLUCAO_MALHA)
    )
    
    lons = df['longitude'].values
    lats = df['latitude'].values
    temps = df['temperatura'].values
    
    distancias = np.sqrt((lons[:, None, None] - grade_lon)**2 + (lats[:, None, None] - grade_lat)**2)
    distancias = np.where(distancias < 1e-10, 1e-10, distancias)
    pesos = 1.0 / (distancias ** 2)
    
    temp_matriz = np.sum(pesos * temps[:, None, None], axis=0) / np.sum(pesos, axis=0)

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
# INTERFACE PRINCIPAL (DASHBOARD)
# ==========================================
def main():
    esconder_menu = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        </style>
        """
    st.markdown(esconder_menu, unsafe_allow_html=True)
    
    st.title("🌡️ Monitoramento Microclimático")
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_imagem = os.path.join(pasta_atual, "LIGA.png")
    col_esq, col_meio, col_dir = st.columns([1, 1, 1])
    with col_meio:
        st.image(caminho_imagem, width=200)
    st.markdown("Dashboard de interpolação térmica em tempo real da UFRRJ.")
    
    df_sensores = obter_dados_sensores()
    if df_sensores.empty:
        st.stop()

    # --- SIMULAÇÃO DE GPS DO USUÁRIO ---
    # Coordenadas fixas fingindo ser o celular de quem abriu o site
    lat_atual = -22.7694
    lon_atual = -43.6875
    
    # Calcula a temperatura exata onde o usuário está pisando
    temp_local_exata = estimar_temp_idw(lon_atual, lat_atual, df_sensores)

    # Exibe a temperatura local em destaque absoluto
    st.success("📍 Sua localização foi obtida com sucesso.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            label="Temperatura na sua posição exata", 
            value=f"{temp_local_exata:.2f} °C",
        )
    st.markdown("---")

    # Controles de visualização do Mapa
    tipo_mapa = st.radio(
        "Selecione a Camada de Visualização:", 
        ["Mapa Geral", "Mapa de Calor"],
        horizontal=True
    )

    mapa_base = folium.Map(location=MAPA_CENTRO, zoom_start=ZOOM_INICIAL)

    # 1. Desenha os sensores fixos
    for _, linha in df_sensores.iterrows():
        folium.CircleMarker(
            location=[linha['latitude'], linha['longitude']],
            radius=6, color="#333333", fill=True, fill_color="#ff4444", fill_opacity=1,
            tooltip=f"Sensor {linha['sensor_id']} | {linha['temperatura']}°C"
        ).add_to(mapa_base)

    # 2. Desenha a superfície de calor se ativada
    if tipo_mapa == "Mapa de Calor":
        limites_terreno = (-43.6882, -43.6868, -22.7700, -22.7688)
        camada_imagem = gerar_camada_isolinhas(df_sensores, limites_terreno)

        raster_layers.ImageOverlay(
            image=camada_imagem,
            bounds=[[-22.7700, -43.6882], [-22.7688, -43.6868]],
            opacity=0.6,
            interactive=False,
            cross_origin=False,
            zindex=1
        ).add_to(mapa_base)

    # 3. Desenha o usuário no mapa com a temperatura no balãozinho
    folium.Marker(
        location=[lat_atual, lon_atual],
        popup=folium.Popup(f"<b>Você está aqui</b><br>Temp: {temp_local_exata:.2f}°C", max_width=200),
        icon=folium.Icon(color="blue", icon="user"),
        tooltip="Sua Posição"
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
