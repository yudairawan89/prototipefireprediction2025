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
    page_title="Smart Fire Prediction HSEL",
    page_icon="🔥",
    layout="wide"
)

# =========================
# Theming (Light/Dark Toggle)
# =========================
if "dark" not in st.session_state:
    st.session_state.dark = False

with st.sidebar:
    st.markdown("### ⚙️ Tampilan")
    st.session_state.dark = st.toggle("Dark mode", value=st.session_state.dark)
    st.caption("Ubah tema agar kontras & aksen warna menyesuaikan.")

# Palet warna dinamis
if st.session_state.dark:
    BG = "#0B1220"
    PANEL = "rgba(255,255,255,0.06)"
    TEXT = "#E8EEF9"
    MUTED = "#9FB0C8"
    ACCENT = "#7C3AED"   # indigo/violet
    ACCENT2 = "#10B981"  # emerald
    BORDER = "rgba(255,255,255,0.12)"
    SHADOW = "0 10px 30px rgba(0,0,0,0.55)"
    GRAD = "linear-gradient(135deg, #1E293B 0%, #0B1220 100%)"
else:
    BG = "#F6F7FB"
    PANEL = "rgba(255,255,255,0.72)"
    TEXT = "#0F172A"
    MUTED = "#475569"
    ACCENT = "#4F46E5"   # indigo
    ACCENT2 = "#059669"  # emerald
    BORDER = "rgba(15,23,42,0.08)"
    SHADOW = "0 10px 24px rgba(2,6,23,0.08)"
    GRAD = "linear-gradient(135deg, #EEF2FF 0%, #E6FFFB 100%)"

# =========================
# Global Styles
# =========================
st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: {BG};
        color: {TEXT};
    }}
    .glass {{
        background: {PANEL};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {BORDER};
        border-radius: 18px;
        box-shadow: {SHADOW};
    }}
    .hero {{
        background: {GRAD};
        border-radius: 18px;
        padding: 22px 26px;
        position: relative;
        overflow: hidden;
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
    }}
    .section-title {{
        font-weight: 800; font-size: 18px; margin: 4px 0 12px 0;
    }}
    /* Button style */
    .link-btn {{
        padding: 8px 16px; border-radius: 10px; text-decoration: none;
        color: white; background: {ACCENT};
        border: 1px solid transparent;
        transition: transform .05s ease;
    }}
    .link-btn:hover {{ transform: translateY(-1px); }}
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
    "Low / Rendah":   ("#1E3A8A", "#DBEAFE"),  # text, bg
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
# HERO HEADER
# =========================
st_autorefresh(interval=7000, key="refresh_realtime")

hero_l, hero_m, hero_r = st.columns([1.1, 7.0, 2.2])
with hero_l:
    st.markdown('<div class="hero glass">', unsafe_allow_html=True)
    st.image("logo.png", width=86)
    st.markdown('</div>', unsafe_allow_html=True)

with hero_m:
    st.markdown('<div class="hero glass">', unsafe_allow_html=True)
    st.markdown(f"""
        <h1 class="title">Smart Fire Prediction HSEL</h1>
        <p class="sub muted">
            Prediksi <b>risiko kebakaran hutan</b> secara real-time berbasis
            <b>Hybrid Stacking Ensemble Learning</b> dengan data sensor lingkungan.
            Aksen warna: <span style="color:{ACCENT}">Indigo</span> & <span style="color:{ACCENT2}">Emerald</span>.
        </p>
        <a class="link-btn" href="https://docs.google.com/spreadsheets/d/1epkIp2U1okjCfXOoz_bkgey4kYa30EtmWlLB6c_911Y/edit?gid=0#gid=0" target="_blank">Data Cloud</a>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with hero_r:
    st.markdown('<div class="hero glass">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.image("logo.png", use_container_width=True)
    with c2:
        st.image("upi.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# REALTIME PREDICTION
# =========================
df = load_data()
container = st.container()
with container:
    box = st.container()
    with box:
        st.markdown('<div class="glass" style="padding:18px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Hasil Prediksi Terkini</div>', unsafe_allow_html=True)

        if df is None or df.empty:
            st.warning("Data belum tersedia atau kosong di Google Sheets.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Rename kolom
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

            # ------------------ Metric Cards ------------------
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("🌡 Suhu (°C)", f"{last_num[fitur[0]]:.1f}")
            m2.metric("💧 RH (%)", f"{last_num[fitur[1]]:.1f}")
            m3.metric("🌧 Curah (mm)", f"{last_num[fitur[2]]:.1f}")
            m4.metric("💨 Angin (m/s)", f"{last_num[fitur[3]]:.1f}")
            m5.metric("🪴 Tanah (%)", f"{last_num[fitur[4]]:.1f}")
            m6.metric("🔥 Risiko", risk_label)

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------ Risk Badge + Last Updated ------------------
            col_a, col_b = st.columns([1.2, 1.2])
            with col_a:
                st.markdown(f"""
                    <div class="metric-card glass">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span class="chip" style="background:{risk_bg}; color:{risk_text}">🔥 {risk_label}</span>
                            <span class="muted">Terakhir diperbarui: <b>{hari}, {tanggal}</b></span>
                        </div>
                        <div class="muted" style="margin-top:8px;">
                            Lokasi: <b>Pekanbaru</b> · Koordinat: <code>-0.5071, 101.4478</code>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with col_b:
                # Map card
                st.markdown('<div class="metric-card glass">', unsafe_allow_html=True)
                st.markdown('<div class="muted" style="margin-bottom:6px;">Peta Prediksi</div>', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# INFO SECTIONS (Accordion)
# =========================
left, right = st.columns(2)
with left:
    with st.expander("ℹ️ Tentang Model", expanded=False):
        st.markdown("""
        **HSEL (Hybrid Stacking Ensemble Learning)** menggabungkan beberapa algoritma
        pembelajaran mesin dengan *stacked generalization* dan optimasi hyperparameter.
        Sistem menerima input: **Suhu, RH, Curah Hujan, Angin, Kelembaban Tanah** dari IoT.
        Output berupa level risiko: *Low, Moderate, High, Very High*.
        """)

with right:
    with st.expander("📊 Cara Membaca Level Risiko", expanded=False):
        st.markdown("""
        - **Low**: kondisi aman, potensi kecil, pemantauan rutin.
        - **Moderate**: waspada, lakukan patroli berkala.
        - **High**: siaga, siapkan sumber daya pemadaman.
        - **Very High**: kondisi kritis, lakukan mitigasi segera.
        """)

# =========================
# Risk Legend (ringkas, modern)
# =========================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass" style="padding:16px;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Legenda Tingkat Risiko</div>', unsafe_allow_html=True)

legend_cols = st.columns(4)
legend_data = [
    ("Low / Rendah", "#DBEAFE", "#1E3A8A", "Intensitas rendah, mudah dikendalikan."),
    ("Moderate / Sedang", "#D1FAE5", "#064E3B", "Masih dapat dikendalikan."),
    ("High / Tinggi", "#FFEDD5", "#7C2D12", "Sulit dikendalikan."),
    ("Very High / Sangat Tinggi", "#FEE2E2", "#7F1D1D", "Sangat sulit dikendalikan."),
]
for (lab, bgc, tc, desc), col in zip(legend_data, legend_cols):
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
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
year = datetime.now().year
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="glass" style="padding:14px; text-align:center;">
  <span class="muted">© {year} Smart Fire Prediction HSEL · Crafted with ❤ · Theme: Indigo & Emerald</span>
</div>
""", unsafe_allow_html=True)
