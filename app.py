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
from streamlit_geolocation import streamlit_geolocation

# ==========================================
# CONFIGURAÇÕES E CONSTANTES
# ==========================================
st.set_page_config(page_title="Monitoramento Geociências", page_icon="🌍", layout="centered")

# Delimitação do Prédio de Geociências
COORDENADAS_CAMPUS = [
    (-43.683322, -22.780782),
    (-43.683864, -22.781685),
    (-43.682205, -22.782115),
    (-43.682452, -22.780999)
]
POLIGONO_GEOCIENCIAS = Polygon(COORDENADAS_CAMPUS)
# Parâmetros de Renderização do Mapa
MAPA_CENTRO = [-22.781448, -43.683034] 
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
        # ATENÇÃO: Adicionamos a 'umidade' na busca do banco de dados
        query = """
            SELECT sensor_id, latitude, longitude, temperatura, umidade, data_hora 
            FROM medicoes 
            ORDER BY data_hora DESC 
            LIMIT 4
        """
        return pd.read_sql_query(query, engine)
    except SQLAlchemyError as e:
        logging.error(f"Falha na conexão: {e}")
        st.error("Serviço temporariamente indisponível. Falha na conexão com o banco.")
        return pd.DataFrame()

def estimar_valor_idw(lon_alvo: float, lat_alvo: float, df: pd.DataFrame, coluna_valor: str, power: int = 2) -> float:
    """Calcula o valor exato (temperatura ou umidade) usando o Inverso do Quadrado da Distância."""
    lons = df['longitude'].values
    lats = df['latitude'].values
    valores = df[coluna_valor].values
    
    distancias = np.sqrt((lons - lon_alvo)**2 + (lats - lat_alvo)**2)
    distancias = np.where(distancias < 1e-10, 1e-10, distancias) 
    
    pesos = 1.0 / (distancias ** power)
    return float(np.sum(pesos * valores) / np.sum(pesos))

def gerar_camada_isolinhas(df: pd.DataFrame, limites: Tuple[float, float, float, float], coluna_valor: str, mapa_cores: str) -> str:
    """Gera o mapa de interpolação científico e o converte para imagem."""
    min_lon, max_lon, min_lat, max_lat = limites

    grade_lon, grade_lat = np.meshgrid(
        np.linspace(min_lon, max_lon, RESOLUCAO_MALHA),
        np.linspace(min_lat, max_lat, RESOLUCAO_MALHA)
    )
    
    lons = df['longitude'].values
    lats = df['latitude'].values
    valores = df[coluna_valor].values
    
    distancias = np.sqrt((lons[:, None, None] - grade_lon)**2 + (lats[:, None, None] - grade_lat)**2)
    distancias = np.where(distancias < 1e-10, 1e-10, distancias)
    pesos = 1.0 / (distancias ** 2)
    
    matriz_valores = np.sum(pesos * valores[:, None, None], axis=0) / np.sum(pesos, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off') 
    fig.patch.set_alpha(0.0) 
    ax.patch.set_alpha(0.0)

    # Usa o mapa_cores (ex: 'coolwarm' para calor, 'Blues' para umidade)
    ax.contourf(grade_lon, grade_lat, matriz_valores, levels=15, cmap=mapa_cores, alpha=0.5)
    ax.contour(grade_lon, grade_lat, matriz_valores, levels=15, colors='black', linewidths=0.5, alpha=0.5)

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
        </style>
        """
    st.markdown(esconder_menu, unsafe_allow_html=True)
    
    st.title("Monitoramento Microclimático")
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_imagem = os.path.join(pasta_atual, "LIGA.png")
    
    col_esq, col_meio, col_dir = st.columns([1, 1, 1])
    with col_meio:
        st.image(caminho_imagem, width=200)
    st.markdown("Dashboard de interpolação térmica e de umidade em tempo real da UFRRJ.")
    
    df_sensores = obter_dados_sensores()
    if df_sensores.empty or 'umidade' not in df_sensores.columns:
        st.warning("Aguardando dados de Temperatura e Umidade do banco de dados...")
        st.stop()

    # --- CAPTURA E VALIDAÇÃO DE GPS ---
    st.write("📍 Obtendo sua localização...")
    localizacao_gps = streamlit_geolocation()

    # Define o Centro do Pátio como ponto de partida (Plano B seguro)
    lat_atual = -22.781448
    lon_atual = -43.683034
    titulo_local = "Centro da Área de Estudo"

    if localizacao_gps['latitude'] is not None and localizacao_gps['longitude'] is not None:
        ponto_usuario = Point(localizacao_gps['longitude'], localizacao_gps['latitude'])
        
        if POLIGONO_GEOCIENCIAS.contains(ponto_usuario):
            lat_atual = localizacao_gps['latitude']
            lon_atual = localizacao_gps['longitude']
            titulo_local = "Sua posição exata"
            st.success("✅ GPS conectado: Você está dentro da área do projeto.")
        else:
            st.warning("⚠️ Você está fora da área do Prédio de Geociências. Exibindo dados do centro do pátio.")
    else:
        st.info("ℹ️ Usando localização padrão. Ative o GPS para precisão local.")

    # --- CÁLCULO DAS MÉTRICAS ---
    # Agora calculamos as duas variáveis passando os nomes das colunas
    temp_local_exata = estimar_valor_idw(lon_atual, lat_atual, df_sensores, 'temperatura')
    umid_local_exata = estimar_valor_idw(lon_atual, lat_atual, df_sensores, 'umidade')

    # Exibe as duas métricas lado a lado
    st.markdown(f"**Local de leitura:** {titulo_local}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🌡️ Temperatura", value=f"{temp_local_exata:.2f} °C")
    with col2:
        st.metric(label="💧 Umidade Relativa", value=f"{umid_local_exata:.2f} %")
        
    st.markdown("---")

    # Controles de visualização do Mapa (Agora com 3 opções!)
    tipo_mapa = st.radio(
        "Selecione a Camada de Visualização:", 
        ["Mapa Geral", "Mapa de Calor", "Mapa de Umidade"],
        horizontal=True
    )

    mapa_base = folium.Map(location=MAPA_CENTRO, zoom_start=ZOOM_INICIAL)

    # 1. Desenha os sensores fixos (agora mostrando Temp e Umidade no balão)
    for _, linha in df_sensores.iterrows():
        folium.CircleMarker(
            location=[linha['latitude'], linha['longitude']],
            radius=6, color="#333333", fill=True, fill_color="#ff4444", fill_opacity=1,
            tooltip=f"Sensor {linha['sensor_id']} | Temp: {linha['temperatura']}°C | Umid: {linha['umidade']}%"
        ).add_to(mapa_base)

    # 2. Desenha a superfície escolhida (Calor ou Umidade)
    if tipo_mapa in ["Mapa de Calor", "Mapa de Umidade"]:
        limites_terreno = (-43.684000, -43.682000, -22.782500, -22.780500)
        
        # Decide qual coluna e qual paleta de cores usar
        if tipo_mapa == "Mapa de Calor":
            coluna_alvo = 'temperatura'
            paleta_cores = 'coolwarm'
        else: # Mapa de Umidade
            coluna_alvo = 'umidade'
            paleta_cores = 'Blues' # Tons de azul para a umidade!

        camada_imagem = gerar_camada_isolinhas(df_sensores, limites_terreno, coluna_alvo, paleta_cores)

        raster_layers.ImageOverlay(
            image=camada_imagem,
            bounds=[[-22.782500, -43.684000], [-22.780500, -43.682000]],
            opacity=0.6,
            interactive=False,
            cross_origin=False,
            zindex=1
        ).add_to(mapa_base)

    # 3. Desenha o usuário no mapa com os dois dados
    folium.Marker(
        location=[lat_atual, lon_atual],
        popup=folium.Popup(f"<b>{titulo_local}</b><br>Temp: {temp_local_exata:.2f}°C<br>Umid: {umid_local_exata:.2f}%", max_width=200),
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
