"""
Sistema de Monitoramento e Interpolação Climática - UFRRJ (Geociências)
Desenvolvido com Streamlit, GeoPandas, SciPy e Folium.
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
from scipy.interpolate import Rbf
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

def calcular_interpolacao_rbf(df: pd.DataFrame, fun_type: str = 'cubic') -> Rbf:
    """
    Gera o modelo matemático de interpolação espacial com base nos sensores.
    """
    return Rbf(
        df['longitude'].values, 
        df['latitude'].values, 
        df['temperatura'].values, 
        function=fun_type
    )

def gerar_camada_isolinhas(modelo_rbf: Rbf, limites: Tuple[float, float, float, float]) -> str:
    """
    Gera a sobreposição de contornos (isopletas) via Matplotlib e converte para Base64.
    """
    min_lon, max_lon, min_lat, max_lat = limites

    # Geração da malha espacial
    grade_lon, grade_lat = np.meshgrid(
        np.linspace(min_lon, max_lon, RESOLUCAO_MALHA),
        np.linspace(min_lat, max_lat, RESOLUCAO_MALHA)
    )
    
    # Aplicação do modelo preditivo
    temp_matriz = modelo_rbf(grade_lon.flatten(), grade_lat.flatten()).reshape(RESOLUCAO_MALHA, RESOLUCAO_MALHA)

    # Renderização científica
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off') 
    fig.patch.set_alpha(0.0) 
    ax.patch.set_alpha(0.0)

    ax.contourf(grade_lon, grade_lat, temp_matriz, levels=15, cmap='coolwarm', alpha=0.5)
    ax.contour(grade_lon, grade_lat, temp_matriz, levels=15, colors='black', linewidths=0.5, alpha=0.5)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # Processamento em memória
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
    st.title("Mapa de Calor")
    st.markdown("---")

    df_sensores = obter_dados_sensores()
    
    if df_sensores.empty:
        st.stop()

    # Layout em colunas para os controles
    col_controle, col_mapa = st.columns([1, 3])

    with col_controle:
        st.subheader("Simulador de Posicionamento")
        st.markdown("Ajuste as coordenadas para estimar a temperatura local.")
        
        lon_usuario = st.slider("Longitude", -43.6890, -43.6860, -43.6875, format="%.5f")
        lat_usuario = st.slider("Latitude", -22.7710, -22.7680, -22.7694, format="%.5f")
        ponto_usuario = Point(lon_usuario, lat_usuario)

        tipo_mapa = st.radio(
            "Camada de Visualização:", 
            ["Sensores e Posição Atual", "Superfície Interpolada (Isolinhas)"]
        )

        st.markdown("---")
        
        # Processamento e Validação da Cerca Virtual
        if POLIGONO_GEOCIENCIAS.contains(ponto_usuario):
            modelo_rbf = calcular_interpolacao_rbf(df_sensores)
            temp_estimada = float(modelo_rbf(lon_usuario, lat_usuario))
            
            st.success("✅ Coordenada válida (Área Interna)")
            st.metric(label="Temperatura Estimada", value=f"{temp_estimada:.2f} °C")
        else:
            temp_estimada = None
            st.error("❌ Fora da área de cobertura do projeto.")

    with col_mapa:
        mapa_base = folium.Map(location=MAPA_CENTRO, zoom_start=ZOOM_INICIAL)

        # Adição dos sensores de referência em ambos os mapas
        for _, linha in df_sensores.iterrows():
            folium.CircleMarker(
                location=[linha['latitude'], linha['longitude']],
                radius=6, color="#333333", fill=True, fill_color="#ff4444", fill_opacity=1,
                tooltip=f"Sensor {linha['sensor_id']} | {linha['temperatura']}°C"
            ).add_to(mapa_base)

        if tipo_mapa == "📍 Sensores e Posição Atual":
            if temp_estimada is not None:
                folium.Marker(
                    location=[lat_usuario, lon_usuario],
                    popup=folium.Popup(f"<b>Temperatura:</b> {temp_estimada:.1f}°C", max_width=200),
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(mapa_base)
            
            st_folium(mapa_base, height=500, use_container_width=True, key="mapa_posicao")

        else:
            # Cálculo dos limites geográficos do polígono para a renderização
            min_lon, max_lon = -43.6882, -43.6868
            min_lat, max_lat = -22.7700, -22.7688
            limites_terreno = (min_lon, max_lon, min_lat, max_lat)

            # Geração da camada científica
            modelo_rbf = calcular_interpolacao_rbf(df_sensores, fun_type='cubic')
            camada_imagem = gerar_camada_isolinhas(modelo_rbf, limites_terreno)

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
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(mapa_base)

            st_folium(mapa_base, height=500, use_container_width=True, key="mapa_cientifico")

if __name__ == "__main__":
    main()
