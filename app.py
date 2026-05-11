import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon
from scipy.interpolate import Rbf

st.title("🌡️ Mapa de Calor Geociências - Nuvem")

# 1. CONECTA NO SUPABASE
@st.cache_data(ttl=10) # Atualiza a cada 10 segundos
def puxar_dados():
    # Ele vai ler a senha escondida que configuraremos no Streamlit
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

st.write("Dados no Banco de Dados (Supabase):")
st.dataframe(df_sensores)
