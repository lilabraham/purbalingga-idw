from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ------------------ Konfigurasi dasar halaman ------------------
st.set_page_config(
    page_title="Peta Kerawanan Kriminalitas Purbalingga 2024 (IDW)",
    layout="wide",
)

# ------------------ Styling global: full map look ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Hilangkan toolbar & footer Streamlit */
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
        position: fixed;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #020617;
        color: #E5E7EB;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif;
        padding: 0;
        margin: 0;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding: 0;
        margin: 0;
    }

    .main .block-container {
        padding-top: 0.4rem;
        padding-bottom: 0.0rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
        max-width: 100%;
    }

    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        gap: 1rem;
    }

    .top-title {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #F9FAFB;
    }

    .top-subtitle {
        font-size: 0.9rem;
        color: #9CA3AF;
        margin-top: 0.15rem;
    }

    /* Styling iframe map (Folium HTML yang di-embed) */
    [data-testid="stIFrame"] iframe {
        border-radius: 18px;
        border: 1px solid #111827;
        box-shadow: 0 26px 60px rgba(0,0,0,0.85);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Path ke file peta ------------------
BASE = Path(__file__).resolve().parent
MAP_HTML = BASE / "outputs" / "html" / "06_idw_surface.html"

# ------------------ Header + "hamburger" info ------------------
col_title, col_info = st.columns([5, 2])

with col_title:
    st.markdown(
        """
        <div class="top-bar">
            <div>
                <div class="top-title">
                    PETA KERAWANAN KRIMINALITAS KABUPATEN PURBALINGGA 2024
                </div>
                <div class="top-subtitle">
                    Permukaan IDW (grid ~100 m), laju per 100.000 penduduk,
                    hotspot P95 &amp; layer desa.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_info:
    with st.expander("☰ Info singkat proyek", expanded=False):
        st.markdown(
            """
            **Ringkasan proyek**

            - **Lokasi & waktu**: Kabupaten Purbalingga, tahun 2024.  
            - **Data**: 34 kejadian (setelah kurasi) pada 28 desa/kelurahan,
              dengan penduduk 2024 per desa.  
            - **Indikator utama**: laju kejadian kriminal per 100.000 penduduk.  
            - **Metode**: Inverse Distance Weighting (IDW) berbasis centroid desa,
              grid ~100 m, parameter hasil LOOCV \\(p = 2, k = 12.  
            - **Evaluasi LOOCV**:
              RMSE ≈ 14,80; MAE ≈ 10,00; MAPE ≈ 45,78%.  
            - **Cara membaca peta**:
              gunakan panel layer di kanan atas peta untuk menyalakan/mematikan
              permukaan kerawanan, hotspot P95 ≥ 39,36, batas desa, serta titik
              desa (tooltip menampilkan laju, jumlah kasus, dan penduduk).
            """
        )

# ------------------ Embed peta Folium (dibuat setinggi mungkin) ------------------
st.markdown("")  # sedikit spasi tipis di bawah header

if MAP_HTML.exists():
    html = MAP_HTML.read_text(encoding="utf-8")
    # Tinggi besar supaya peta hampir full screen.
    # Kalau di layar kamu masih ada ruang kosong / terlalu tinggi,
    # cukup ubah angkanya (misal 800, 900, 950).
    components.html(html, height=900, scrolling=False)
else:
    st.error(f"Tidak menemukan file peta: {MAP_HTML}")
    st.write(
        "Pastikan file sudah dibuat oleh skrip `scripts/11_webmap_final.py` "
        "dan struktur folder `outputs/html/06_idw_surface.html` sesuai."
    )
