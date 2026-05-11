import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon
from scipy.interpolate import Rbf
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import numpy as np

st.title("Mapa de Calor Geociências")

# 1. CONECTA NO SUPABASE
@st.cache_data(ttl=10) # Atualiza a cada 10 segundos
def puxar_dados():
   
    engine = create_engine(st.secrets["DB_URL"])
    query = "SELECT sensor_id, latitude, longitude, temperatura, data_hora FROM medicoes ORDER BY data_hora DESC LIMIT 4"
    return pd.read_sql_query(query, engine)

df_sensores = puxar_dados()

# 2. CERCA VIRTUAL
coords_geo = [(-43.6882, -22.7688), (-43.6868, -22.7688), (-43.6868, -22.7700), (-43.6882, -22.7700)]
poligono_geo = Polygon(coords_geo)

# 3. TESTE DE GPS
st.write("Mova o slider para simular o aluno andando pelo pátio:")
lon_usuario = st.slider("Longitude", -43.6890, -43.6860, -43.6875, format="%.5f")
lat_usuario = st.slider("Latitude", -22.7710, -22.7680, -22.7694, format="%.5f")
ponto_usuario = Point(lon_usuario, lat_usuario)

# 4. CÁLCULO
if poligono_geo.contains(ponto_usuario):
    st.success("Na área! Calculando...")
    x = df_sensores['longitude'].values
    y = df_sensores['latitude'].values
    z = df_sensores['temperatura'].values
    
    funcao_calor = Rbf(x, y, z, function='linear')
    temp_estimada = funcao_calor(lon_usuario, lat_usuario)
    st.metric(label="Temperatura Estimada", value=f"{temp_estimada:.2f} °C")
else:
    st.error("Fora da área de cobertura.")


# 5. DESENHO DO MAPA VISUAL
st.subheader("🗺️ Visualização Espacial")

# Trocamos as "tabs" por "radio" para evitar o bug de mapas em elementos ocultos
tipo_mapa = st.radio("Escolha a visualização:", ["📍 Mapa de Posição (Sliders)", "🌡️ Mapa de Calor Interpolado"])

if tipo_mapa == "📍 Mapa de Posição (Sliders)":
    st.write("Aqui você vê a sua posição exata baseada nos sliders.")
    mapa_posicao = folium.Map(location=[-22.7694, -43.6875], zoom_start=18)
    
    # Marca os 4 sensores com pontinhos vermelhos
    for index, linha in df_sensores.iterrows():
        folium.CircleMarker(
            location=[linha['latitude'], linha['longitude']],
            radius=6, color="red", fill=True, 
            tooltip=f"Sensor {linha['sensor_id']}: {linha['temperatura']}°C"
        ).add_to(mapa_posicao)

    # Marca o usuário se ele estiver dentro da área
    if poligono_geo.contains(ponto_usuario):
        folium.Marker(
            location=[lat_usuario, lon_usuario],
            popup=f"Sua Temp: {temp_estimada:.1f}°C",
            icon=folium.Icon(color="blue", icon="user"),
        ).add_to(mapa_posicao)

    # A 'key' única é essencial para o Streamlit não confundir os mapas
    st_folium(mapa_posicao, height=400, use_container_width=True, key="mapa_pos")

else:
    st.write("Superfície contínua de temperatura calculada pelo modelo matemático.")
    mapa_calor = folium.Map(location=[-22.7694, -43.6875], zoom_start=18)

    # Limites do prédio de geociências
    min_lon, max_lon = -43.6882, -43.6868
    min_lat, max_lat = -22.7700, -22.7688

    # Criação da malha com 2500 pontos invisíveis (50x50)
    grade_lon, grade_lat = np.meshgrid(
        np.linspace(min_lon, max_lon, 50),
        np.linspace(min_lat, max_lat, 50)
    )
    
    lon_achapada = grade_lon.flatten()
    lat_achapada = grade_lat.flatten()
    temp_achapada = funcao_calor(lon_achapada, lat_achapada)

    # Filtra os pontos que estão dentro do terreno e converte pro formato seguro
    dados_grade = []
    for ln, lt, tp in zip(lon_achapada, lat_achapada, temp_achapada):
        if poligono_geo.contains(Point(float(ln), float(lt))):
            # O truque de mestre: float() limpa a formatação NumPy que quebra o mapa
            dados_grade.append([float(lt), float(ln), float(tp)]) 

    # Aplica o HeatMap denso
    from folium.plugins import HeatMap
    HeatMap(dados_grade, radius=18, blur=12, min_opacity=0.4).add_to(mapa_calor)

    # Marca onde estão os sensores reais por cima do calor para referência
    for index, linha in df_sensores.iterrows():
        folium.Marker(
            location=[linha['latitude'], linha['longitude']],
            icon=folium.Icon(color="black", icon="info-sign"),
            tooltip=f"Sensor {linha['sensor_id']} (Real)"
        ).add_to(mapa_calor)

    # Outra 'key' única
    st_folium(mapa_calor, height=400, use_container_width=True, key="mapa_calor")
