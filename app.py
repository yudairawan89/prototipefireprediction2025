import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from streamlit_folium import folium_static
import folium

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="UHTP Smart Fire Prediction",
    page_icon="🔥",
    layout="wide"
)

# =========================
# Static Theme (Light)
# =========================
BG = "#F6F7FB"
PANEL = "rgba(255,255,255,0.82)"
TEXT = "#0F172A"
MUTED = "#475569"
ACCENT = "#4F46E5"   # indigo
ACCENT2 = "#059669"  # emerald
BORDER = "rgba(15,23,42,0.08)"
SHADOW = "0 10px 24px rgba(2,6,23,0.08)"

st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: {BG};
        color: {TEXT};
    }}
    .chip {{
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        border: 1px solid {BORDER};
    }}
    .muted {{ color: {MUTED}; }}
    .title {{
        margin: 0; padding: 0;
        font-size: 32px; line-height: 1.2; font-weight: 800;
    }}
    .sub {{
        margin-top: 6px;
        font-size: 15px; line-height: 1.6;
    }}
    .metric-card {{
        padding: 16px; border-radius: 16px; border: 1px solid {BORDER};
        background: rgba(255,255,255,0.55);
    }}
    /* Tombol Data Cloud */
    .link-btn {{
        display: inline-block;
        padding: 10px 16px; border-radius: 12px; text-decoration: none;
        color: #ffffff !important; background: linear-gradient(135deg,{ACCENT} 0%, {ACCENT2} 100%);
        border: 1px solid transparent;
        box-shadow: {SHADOW};
        transition: transform .05s ease, filter .2s ease;
        font-weight: 600;
    }}
    .link-btn:hover {{ transform: translateY(-1px); filter: brightness(1.02); }}

    /* ====== Card reusable (frame) ====== */
    .panel-card {{
        border: 1px solid {BORDER};
        border-radius: 14px;
        overflow: hidden;
        box-shadow: {SHADOW};
        background: rgba(255,255,255,0.82);
    }}
    .title-bar {{
        padding: 10px 14px;
        font-weight: 800;
        color: {MUTED};
        border-bottom: 1px solid {BORDER};
        background: rgba(255,255,255,0.55);
        display: flex; align-items: center; gap: 8px;
    }}
    .card-content {{ padding: 16px; }}

    /* Map card sinkron dengan panel-card */
    .map-card {{ border: 1px solid {BORDER}; border-radius: 14px; overflow: hidden;
                box-shadow: {SHADOW}; background: rgba(255,255,255,0.55); }}
    .map-title-bar {{ padding: 10px 14px; font-weight: 800; color: {MUTED};
                     border-bottom: 1px solid {BORDER}; background: rgba(255,255,255,0.55); }}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def convert_day_to_indonesian(day_name):
    return {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {
        'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
        'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
        'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
        'November': 'November', 'December': 'Desember'
    }.get(month_name, month_name)

def convert_to_label(pred):
    return {
        0: "Low / Rendah",
        1: "Moderate / Sedang",
        2: "High / Tinggi",
        3: "Very High / Sangat Tinggi"
    }.get(pred, "Unknown")

risk_styles = {
    "Low / Rendah":   ("#1E3A8A", "#DBEAFE"),
    "Moderate / Sedang": ("#064E3B", "#D1FAE5"),
    "High / Tinggi":  ("#7C2D12", "#FFEDD5"),
    "Very High / Sangat Tinggi": ("#7F1D1D", "#FEE2E2")
}

# =========================
# Load Model & Scaler
# =========================
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# =========================
# Data loader (Sheets CSV)
# =========================
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1epkIp2U1okjCfXOoz_bkgey4kYa30EtmWlLB6c_911Y/export?format=csv"
    return pd.read_csv(url)

# =========================
# AUTOREFRESH
# =========================
st_autorefresh(interval=7000, key="refresh_realtime")

# =========================
# HEADER — di dalam frame TANPA title bar
# =========================
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="card-content">', unsafe_allow_html=True)

hdr_l, hdr_m, hdr_r = st.columns([1.1, 7.0, 2.2])
with hdr_l:
    st.image("uhts.png")
with hdr_m:
    st.markdown("""
        <h1 class="title">UHTP Smart Fire Prediction</h1>
        <p class="sub muted">
            Sistem Prediksi <b>tingkat risiko kebakaran hutan</b> real-time berbasis
            <b>Hybrid Machine Learning</b> dari data sensor IoT.
        </p>
        <a class="link-btn" href="https://docs.google.com/spreadsheets/d/1epkIp2U1okjCfXOoz_bkgey4kYa30EtmWlLB6c_911Y/edit?gid=0#gid=0" target="_blank">☁️&nbsp; Data Cloud</a>
    """, unsafe_allow_html=True)
with hdr_r:
    # HANYA logo smft di sisi kanan
    st.image("logo smft.png", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)  # close card-content
st.markdown('</div>', unsafe_allow_html=True)  # close panel-card

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# REALTIME PREDICTION (judul di dalam frame)
# =========================
df = load_data()

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="title-bar">📈 Hasil Prediksi Terkini</div>', unsafe_allow_html=True)
st.markdown('<div class="card-content">', unsafe_allow_html=True)

if df is None or df.empty:
    st.warning("Data belum tersedia atau kosong di Google Sheets.")
else:
    df = df.rename(columns={
        'Timestamp': 'Waktu',
        'Suhu': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan': 'RR: Curah hujan (mm)',
        'Kecepatan Angin': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Permukaan Tanah',
    })

    fitur = [
        'Tavg: Temperatur rata-rata (°C)',
        'RH_avg: Kelembapan rata-rata (%)',
        'RR: Curah hujan (mm)',
        'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembaban Permukaan Tanah'
    ]

    missing = [c for c in fitur + ['Waktu'] if c not in df.columns]
    if missing:
        st.error("Kolom wajib tidak ditemukan di Sheets: " + ", ".join(missing))
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    clean_df = df[fitur].copy()
    for col in fitur:
        clean_df[col] = (
            clean_df[col].astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
            .fillna(0)
        )
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaled_all = scaler.transform(clean_df)
    predictions = [convert_to_label(p) for p in model.predict(scaled_all)]
    df["Prediksi Kebakaran"] = predictions

    last_row = df.iloc[-1]
    last_num = clean_df.iloc[-1]
    waktu = pd.to_datetime(last_row["Waktu"])
    hari = convert_day_to_indonesian(waktu.strftime("%A"))
    bulan = convert_month_to_indonesian(waktu.strftime("%B"))
    tanggal = waktu.strftime(f"%d {bulan} %Y")
    risk_label = last_row["Prediksi Kebakaran"]
    risk_text, risk_bg = risk_styles.get(risk_label, ("#111827", "#E5E7EB"))

    # --------- Metrics (5 saja) ----------
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🌡 Suhu Udara (°C)", f"{last_num[fitur[0]]:.1f}")
    m2.metric("💧 Kelembaban Udara RH (%)", f"{last_num[fitur[1]]:.1f}")
    m3.metric("🌧 Curah Hujan (mm)", f"{last_num[fitur[2]]:.1f}")
    m4.metric("💨 Kecepatan Angin (m/s)", f"{last_num[fitur[3]]:.1f}")
    m5.metric("🪴 Kelembaban Tanah (%)", f"{last_num[fitur[4]]:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------- Ringkasan + Map ----------
    col_a, col_b = st.columns([1.2, 1.2])

    # LEFT COLUMN (summary + 2 DROPDOWNS)
    with col_a:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                    <span class="chip" style="background:{risk_bg}; color:{risk_text}">🔥 {risk_label}</span>
                    <span class="muted">Terakhir diperbarui: <b>{hari}, {tanggal}</b></span>
                </div>
                <div class="muted" style="margin-top:8px;">
                    Lokasi: <b>Pekanbaru</b> · Koordinat:
                    <span style="color:#16A34A;">-0.5071, 101.4478</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # === DROPDOWN 1: Keterangan Aplikasi ===
        with st.expander("ℹ️ Keterangan Aplikasi", expanded=False):
            st.markdown("#### 🧠 Tentang Model")
            st.markdown("""
            **Hybrid Machine Learning** menggabungkan beberapa algoritma
            pembelajaran mesin dengan optimasi hyperparameter.
            Input dari IoT: **Suhu, RH, Curah Hujan, Angin, Kelembaban Tanah**.
            Output: level risiko **Low, Moderate, High, Very High**.
            """)
            st.markdown("---")
            st.markdown("#### 📘 Cara Membaca Level Risiko")
            st.markdown("""
            - **Low** — kondisi aman, pemantauan rutin.
            - **Moderate** — waspada, lakukan patroli berkala.
            - **High** — siaga, siapkan sumber daya pemadaman.
            - **Very High** — kondisi kritis, lakukan mitigasi segera.
            """)

        # === DROPDOWN 2: Legenda Risiko ===
        with st.expander("🏷️ Legenda Tingkat Risiko", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            legend_data = [
                ("Low / Rendah", "#DBEAFE", "#1E3A8A", "Intensitas rendah, mudah dikendalikan."),
                ("Moderate / Sedang", "#D1FAE5", "#064E3B", "Masih dapat dikendalikan."),
                ("High / Tinggi", "#FFEDD5", "#7C2D12", "Sulit dikendalikan."),
                ("Very High / Sangat Tinggi", "#FEE2E2", "#7F1D1D", "Sangat sulit dikendalikan."),
            ]
            for (lab, bgc, tc, desc), col in zip(legend_data, [c1, c2, c3, c4]):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="background:{bgc}; color:{tc};">
                            <b>{lab}</b><br>
                            <span style="color:{tc}; font-size:13px;">{desc}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # RIGHT COLUMN (map)
    with col_b:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        st.markdown('<div class="map-title-bar">🗺️ Peta Prediksi</div>', unsafe_allow_html=True)

        pekanbaru_coords = [-0.5071, 101.4478]
        color_map = {
            "Low / Rendah": "blue",
            "Moderate / Sedang": "green",
            "High / Tinggi": "orange",
            "Very High / Sangat Tinggi": "red"
        }
        marker_color = color_map.get(risk_label, "gray")

        popup_text = folium.Popup(f"""
            <div style='width: 230px; font-size: 13px; line-height: 1.5;'>
            <b>Prediksi:</b> {risk_label}<br>
            <b>Suhu:</b> {last_num[fitur[0]]:.1f} °C<br>
            <b>Kelembapan:</b> {last_num[fitur[1]]:.1f} %<br>
            <b>Curah Hujan:</b> {last_num[fitur[2]]:.1f} mm<br>
            <b>Kecepatan Angin:</b> {last_num[fitur[3]]:.1f} m/s<br>
            <b>Kelembaban Tanah:</b> {last_num[fitur[4]]:.1f} %<br>
            <b>Waktu:</b> {last_row['Waktu']}
            </div>
        """, max_width=250)

        m = folium.Map(location=pekanbaru_coords, zoom_start=11, tiles="CartoDB positron")
        folium.Circle(
            location=pekanbaru_coords, radius=3000,
            color=marker_color, fill=True, fill_color=marker_color, fill_opacity=0.3
        ).add_to(m)
        folium.Marker(
            location=pekanbaru_coords, popup=popup_text,
            icon=folium.Icon(color=marker_color, icon="info-sign")
        ).add_to(m)
        folium_static(m, width=520, height=350)
        st.markdown('</div>', unsafe_allow_html=True)  # close map-card

st.markdown('</div>', unsafe_allow_html=True)  # close card-content
st.markdown('</div>', unsafe_allow_html=True)  # close panel-card

# =========================
# Footer
# =========================
year = datetime.now().year
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="panel-card" style="text-align:center;">
  <div class="card-content">
    <span class="muted">© {year} UHTP Smart Fire Prediction · Dirancang Oleh Tim Dosen UHTP</span>
  </div>
</div>
""", unsafe_allow_html=True)


