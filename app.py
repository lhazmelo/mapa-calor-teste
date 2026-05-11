import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon
from scipy.interpolate import Rbf
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

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
st.subheader("🗺️ Visualização do Prédio de Geociências")

#CENTRALIZAÇÃO DO MAPA
mapa = folium.Map(location=[-22.7694, -43.6875], zoom_start=18)

dados_heatmap = []

#MARCADOR VERMELHO PARA CADA SENSOR
for index, linha in df_sensores.iterrows():
    lat = linha['latitude']
    lon = linha['longitude']
    temp = linha['temperatura']
    
    folium.Marker(
        location=[lat, lon],
        popup=f"Sensor {linha['sensor_id']}: {temp}°C",
        icon=folium.Icon(color="red", icon="fire"),
    ).add_to(mapa)
    
    #CRIA UM PESO TÉRMICO
    dados_heatmap.append([lat, lon, temp])

#CAMADA DE CORES
HeatMap(dados_heatmap, radius=40, blur=25, max_zoom=1).add_to(mapa)

#MOSTRA ONDE O ALUNO ESTÁ NO MAPA
if poligono_geo.contains(ponto_usuario):
    folium.Marker(
        location=[lat_usuario, lon_usuario],
        popup="Você está aqui",
        icon=folium.Icon(color="blue", icon="user"),
    ).add_to(mapa)

#JOGA O MAPA NA TELA
st_folium(mapa, width=700, height=500)
