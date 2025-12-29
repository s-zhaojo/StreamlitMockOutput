import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Reservoir Sedimentation Demo", layout="wide")
st.title("Reservoir Sedimentation Visualization Demo")

st.subheader("3D Reservoir Sedimentation Over Time (Interactive)")

nx, ny = 50, 50
x = np.linspace(0, 100, nx)
y = np.linspace(0, 50, ny)
X, Y = np.meshgrid(x, y)

Z_initial = 20 * np.exp(-((X - 50)**2 + (Y - 25)**2) / 200)

years = list(range(2000, 2025))
sedimentation_rate = 0.3
Z_all_years = []
for i, year in enumerate(years):
    loss_factor = (1 - sedimentation_rate * i / 100)
    Z_all_years.append(Z_initial * loss_factor)
Z_all_years = np.array(Z_all_years)

selected_year = st.slider("Select Year", min_value=years[0], max_value=years[-1], value=years[0])
year_index = selected_year - years[0]
Z_current = Z_all_years[year_index]

fig_3d = go.Figure(data=[go.Surface(
    z=Z_current,
    x=X,
    y=Y,
    colorscale='Viridis',
    cmin=0,
    cmax=np.max(Z_initial),
    colorbar=dict(title='Height (m)')
)])
fig_3d.update_layout(
    title=f"Reservoir 3D Surface - Year {selected_year}",
    scene=dict(
        xaxis_title='Width (m)',
        yaxis_title='Length (m)',
        zaxis_title='Height (m)'
    ),
    width=900,
    height=600
)
st.plotly_chart(fig_3d, use_container_width=True)

total_initial = np.sum(Z_initial)
total_current = np.sum(Z_current)
cumulative_loss = (1 - total_current / total_initial) * 100

st.subheader("Sedimentation Metrics (Current Year)")
c1, c2 = st.columns(2)
c1.metric("Cumulative Loss (%)", f"{cumulative_loss:.2f}")
c2.metric("Remaining Volume (%)", f"{100 - cumulative_loss:.2f}")

st.subheader("Static Reservoir Sedimentation Demo")
reservoir_name = "Lake Demo"
lat_demo, lon_demo = 35.0, -120.0
years_static = list(range(2000, 2025))
sedimentation_rate_static = 0.3
cumulative_loss_static = [sedimentation_rate_static * (y - years_static[0]) for y in years_static]
remaining_capacity_static = [100 - cl for cl in cumulative_loss_static]

st.subheader("Sedimentation Metrics (Example)")
c1, c2, c3 = st.columns(3)
c1.metric("Sedimentation Rate (%/year)", f"{sedimentation_rate_static:.2f}")
c2.metric("Cumulative Capacity Loss (%)", f"{cumulative_loss_static[-1]:.2f}")
c3.metric("Remaining Capacity (million m³)", f"{remaining_capacity_static[-1]:.2f}")

sns.set_style("darkgrid")

st.subheader("Cumulative Capacity Loss Over Time")
fig, ax = plt.subplots(figsize=(10,4))
sns.lineplot(x=years_static, y=cumulative_loss_static, marker='o', color='crimson', ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Loss (%)")
ax.set_title(f"{reservoir_name} Sedimentation Over Time")
st.pyplot(fig)

st.subheader("Remaining Capacity Over Time")
fig2, ax2 = plt.subplots(figsize=(10,4))
sns.lineplot(x=years_static, y=remaining_capacity_static, marker='o', color='green', ax=ax2)
ax2.set_xlabel("Year")
ax2.set_ylabel("Remaining Capacity (million m³)")
ax2.set_title(f"{reservoir_name} Remaining Capacity Over Time")
st.pyplot(fig2)

st.subheader("Sedimentation Data Table (Example)")
df_demo = pd.DataFrame({
    "Year": years_static,
    "Cumulative Loss (%)": cumulative_loss_static,
    "Remaining Capacity (million m³)": remaining_capacity_static
})
st.dataframe(df_demo)
