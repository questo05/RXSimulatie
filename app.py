"""
Dashboard — RX Wachtlijstsimulatie UZ Gasthuisberg
Starten: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulatie_nieuw import (
    RXSimulatie,
    AANKOMST_AMBULANT_STUURBAAR, AANKOMST_AMBULANT_VAST,
    AANKOMST_HOSP, AANKOMST_DAG,
    GESCHIKTHEID, DEFAULT_ZALEN, DEFAULT_TECHNICI, DEFAULT_WACHTKAMER,
    DRUKTE_NIVEAUS, SIM_START_UUR, SIM_EINDE_UUR,
)

st.set_page_config(
    page_title="RX Simulatie — UZ Gasthuisberg",
    page_icon="🏥", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.kpi { background:#111827; border:1px solid #1f2937; border-radius:10px; padding:1rem 1.2rem; margin-bottom:.4rem; }
.kpi-lbl { font-family:'IBM Plex Mono',monospace; font-size:.65rem; color:#6b7280; text-transform:uppercase; letter-spacing:.09em; margin-bottom:.2rem; }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:1.7rem; font-weight:600; color:#f9fafb; line-height:1; }
.kpi-unit { font-size:.78rem; color:#6b7280; margin-left:.2rem; }
.kpi-ok   { color:#34d399; font-size:.72rem; margin-top:.15rem; }
.kpi-warn { color:#fbbf24; font-size:.72rem; margin-top:.15rem; }
.kpi-bad  { color:#f87171; font-size:.72rem; margin-top:.15rem; }
.hdr { font-family:'IBM Plex Mono',monospace; font-size:.68rem; color:#6b7280;
       text-transform:uppercase; letter-spacing:.1em; border-bottom:1px solid #1f2937;
       padding-bottom:.3rem; margin:1.3rem 0 .8rem 0; }
.info-box { background:#1a2035; border-left:3px solid #3b82f6; padding:.6rem .9rem;
            border-radius:0 6px 6px 0; font-size:.78rem; color:#93c5fd; margin-bottom:.5rem; }
.stButton>button { background:#059669; color:white; border:none; border-radius:6px;
                   font-family:'IBM Plex Mono',monospace; font-size:.85rem;
                   padding:.5rem 1rem; width:100%; }
.stButton>button:hover { background:#10b981; }
</style>
""", unsafe_allow_html=True)

T = dict(paper_bgcolor="#111827", plot_bgcolor="#111827",
         font=dict(color="#d1d5db", family="IBM Plex Mono"))
G = dict(gridcolor="#1f2937", zerolinecolor="#1f2937")
ZAAL_KLEUR = {'G5':'#f59e0b','G15':'#a78bfa','G3':'#34d399',
              'G4':'#60a5fa','G2':'#f87171','G1':'#6b7280'}

def kpi(lbl, val, unit='', sig=None):
    s = f'<div class="kpi-{sig[0]}">{sig[1]}</div>' if sig else ''
    return f'<div class="kpi"><div class="kpi-lbl">{lbl}</div><div class="kpi-val">{val}<span class="kpi-unit">{unit}</span></div>{s}</div>'

def uur_label(u: float) -> str:
    """Zet float-uur om naar leesbaar label, bv. 8.5 → '8:30'."""
    h = int(u)
    m = int((u - h) * 60)
    return f"{h}:{m:02d}"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 RX Simulatie")
    st.markdown(f"*UZ Gasthuisberg · {SIM_START_UUR}u–{SIM_EINDE_UUR}u*")
    st.divider()

    st.markdown("### 📦 Infrastructuur")
    alle_zalen    = ['G5','G15','G3','G4','G2','G1']
    actieve_zalen = st.multiselect("Actieve RX-zalen", alle_zalen, DEFAULT_ZALEN,
                                   help="G5=thorax · G15=full-length · G2/G3/G4=gemengd · G1=backup")
    n_technici    = st.slider("Radiotechnici", 1, 12, DEFAULT_TECHNICI,
                              help="Aantal beschikbare technici (1 per onderzoek tegelijk)")
    wk_cap        = st.slider("Wachtkamercapaciteit", 5, 80, DEFAULT_WACHTKAMER,
                              help="Patiënten die wachtkamer vol treffen vertrekken meteen")

    st.markdown("### 📅 Drukte-niveau")
    st.caption("Gebaseerd op dagvolume-percentielen KWS 2021-2025")
    drukte_keuze = st.selectbox("Verwacht dagvolume", list(DRUKTE_NIVEAUS.keys()), index=1)
    schaal       = DRUKTE_NIVEAUS[drukte_keuze]

    st.markdown("### ⚙️ Wachtrij-instellingen")
    fifo         = st.toggle("FIFO-wachtrij", value=True,
                             help="AAN = geen prioriteit. UIT = gehospitaliseerden krijgen lichte voorrang.")
    balk_drempel = st.slider("Balking drempel (min)", 15, 180, 60,
                             help="Patiënt verlaat wachtkamer als wachttijd > X minuten")

    st.divider()

    # ── Tijdsloten ────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Tijdsloten ambulant (zonder afspraak)")
    st.markdown("""
    <div style='font-size:.72rem;color:#6b7280;line-height:1.6;margin-bottom:.6rem'>
    Enkel de <b>14.6% ambulante patiënten zonder andere afspraak</b>
    (gem. ~15/dag) kunnen worden doorverwezen naar tijdsloten.
    De overige 85.4% (met consultatie) komen op hun normale patroon.
    </div>
    """, unsafe_allow_html=True)

    slot_pct     = st.slider("% doorverwezen naar tijdsloten", 0, 100, 0, 5)
    slot_fractie = slot_pct / 100

    st.caption("Selecteer tijdsloten (per halfuur)")

    # Halfuur-rooster als visuele toggle: 7u00 t/m 17u30
    alle_halfuren = [h + m for h in range(SIM_START_UUR, SIM_EINDE_UUR)
                     for m in [0, 0.5]]

    # Toon als raster: uren als rijen, :00 en :30 als kolommen
    if 'tijdsloten' not in st.session_state:
        st.session_state.tijdsloten = set()

    col_hdr1, col_hdr2, col_hdr3 = st.columns([2, 1, 1])
    col_hdr1.caption("Uur")
    col_hdr2.caption(":00")
    col_hdr3.caption(":30")

    for uur in range(SIM_START_UUR, SIM_EINDE_UUR):
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f"<div style='padding-top:.3rem;font-size:.82rem'>{uur}u</div>",
                    unsafe_allow_html=True)
        k_vol  = f"slot_{uur}_0"
        k_half = f"slot_{uur}_5"
        vol  = c2.checkbox("", key=k_vol,  value=(uur     in st.session_state.tijdsloten))
        half = c3.checkbox("", key=k_half, value=((uur+0.5) in st.session_state.tijdsloten))
        if vol:  st.session_state.tijdsloten.add(uur)
        else:    st.session_state.tijdsloten.discard(uur)
        if half: st.session_state.tijdsloten.add(uur + 0.5)
        else:    st.session_state.tijdsloten.discard(uur + 0.5)

    tijdsloten = sorted(st.session_state.tijdsloten)

    st.divider()
    scenario_naam = st.text_input("Scenarionaam", "Baseline")
    seed          = st.number_input("Random seed", 0, 9999, 42)
    run_btn       = st.button("▶ Start simulatie", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}
if 'laatste' not in st.session_state:
    st.session_state.laatste = None

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not actieve_zalen:
        st.error("Selecteer minstens één zaal.")
    else:
        config = {
            'actieve_zalen':   actieve_zalen,
            'n_technici':      n_technici,
            'wachtkamer_cap':  wk_cap,
            'balk_drempel_min': balk_drempel,
            'schaal':          schaal,
            'tijdsloten':      tijdsloten,
            'slot_fractie':    slot_fractie,
            'fifo':            fifo,
        }
        with st.spinner(f"Simulatie '{scenario_naam}' bezig..."):
            res = RXSimulatie(config=config, seed=seed).run()
            res['naam']      = scenario_naam
            res['slot_pct']  = slot_pct
            res['drukte']    = drukte_keuze
            st.session_state.scenarios[scenario_naam] = res
            st.session_state.laatste = scenario_naam
        if 'fout' not in res:
            st.success(f"✓ {res['afgehandeld']} patiënten geholpen, {res['gebalkt']} vertrokken")

# ══════════════════════════════════════════════════════════════════════════════
# HOOFDPAGINA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<h1 style='font-family:IBM Plex Mono;font-size:1.4rem;color:#f9fafb;margin-bottom:0'>
  RX Wachtlijstsimulatie — UZ Gasthuisberg
</h1>
<p style='color:#6b7280;font-size:.85rem;margin-top:.3rem'>
  Radiologie walk-in · Simulatievenster: {SIM_START_UUR}u–{SIM_EINDE_UUR}u · KWS-data 2021-2025
</p>
""", unsafe_allow_html=True)

if not st.session_state.scenarios:
    st.info("👈 Stel parameters in en klik op **Start simulatie**.")

    st.markdown('<div class="hdr">Aankomstprofiel (werkdag 2021-2025, 7u–18u)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Wat toont deze grafiek?</b> Het gemiddeld aantal patiënten dat per uur toekomt op een normale werkdag.
    Ambulant is opgesplitst in <b>zonder consultatie (stuurbaar, ~15/dag)</b> en
    <b>met consultatie (niet-stuurbaar, ~87/dag)</b>.
    </div>
    """, unsafe_allow_html=True)

    uren = list(range(SIM_START_UUR, SIM_EINDE_UUR))
    fig  = go.Figure()
    for profiel, naam, kleur in [
        (AANKOMST_AMBULANT_VAST,      'Ambulant met consultatie (niet-stuurbaar)',   '#60a5fa'),
        (AANKOMST_AMBULANT_STUURBAAR, 'Ambulant zonder consultatie (stuurbaar, ~15/dag)', '#93c5fd'),
        (AANKOMST_HOSP,               'Gehospitaliseerd (~23/dag)',               '#f87171'),
        (AANKOMST_DAG,                'Dagziekenhuis (~6/dag)',                   '#34d399'),
    ]:
        fig.add_trace(go.Bar(
            x=uren, y=[profiel.get(h,0) for h in uren],
            name=naam, marker_color=kleur, opacity=.85,
        ))
    fig.update_layout(
        barmode='stack',
        xaxis=dict(title='Uur van de dag', dtick=1, **G),
        yaxis=dict(title='Gem. aantal toekomende', **G),
        height=340, legend=dict(orientation='h', y=-.28), **T,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

m = st.session_state.scenarios[st.session_state.laatste]
if 'fout' in m:
    st.error(m['fout'])
    st.stop()

df = m['patient_df']
kleuren = m['type_kleuren']

# ── Uitleg wachttijden ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="info-box">
<b>Hoe lezen?</b> De simulatie loopt van <b>{SIM_START_UUR}u tot {SIM_EINDE_UUR}u</b>
op basis van het gemiddeld aankomstprofiel van een werkdag.
Drukte-niveau: <b>{m.get('drukte','–')}</b> (schaal ×{m['config']['schaal']:.2f}).
Totaal gesimuleerd: <b>{m['totaal']} patiënten</b>
({m['afgehandeld']} geholpen, {m['gebalkt']} vertrokken voor beurt).
</div>
""", unsafe_allow_html=True)

# ── KPI's ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hdr">Kernresultaten</div>', unsafe_allow_html=True)
cols = st.columns(6)

w = m['gem_wacht']
sig = ('ok','✓ OK') if w<20 else ('warn','⚠ Verhoogd') if w<40 else ('bad','✗ Hoog')
with cols[0]: st.markdown(kpi('Geholpen', m['afgehandeld']), unsafe_allow_html=True)
with cols[1]: st.markdown(kpi('Vertrokken', m['gebalkt'],
                              sig=('bad',f'↑ {m["balk_rate"]:.1f}%') if m['gebalkt']>0 else None),
                          unsafe_allow_html=True)
with cols[2]: st.markdown(kpi('Gem. wachttijd', f"{w:.1f}", 'min', sig), unsafe_allow_html=True)
with cols[3]: st.markdown(kpi('Mediaan wacht',  f"{m['mediaan_wacht']:.1f}", 'min'), unsafe_allow_html=True)
with cols[4]: st.markdown(kpi('P90 wacht',       f"{m['p90_wacht']:.1f}",    'min'), unsafe_allow_html=True)
cv = m['werkdruk_cv']
sig_cv = ('ok','✓ Vlak') if cv<40 else ('warn','⚠ Piekerig') if cv<60 else ('bad','✗ Hoge piek')
with cols[5]: st.markdown(kpi('Werkdruk CV', f"{cv:.1f}", '%', sig_cv), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Wachttijden", "🏥 Zalen & Bezetting",
    "🎯 Tijdsloten-analyse", "🔀 Scenario-vergelijking",
])

# ── TAB 1: Wachttijden ────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="info-box">
    <b>Wachttijd</b> = tijd tussen aankomst van de patiënt en start van het onderzoek.
    Een wachttijd van 0 min betekent dat een zaal en technicus meteen beschikbaar waren.
    De x-as 'uur van de dag' toont het aankomstuur van de patiënt (7 = 7u00–8u00).
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for lbl, kleur in kleuren.items():
            sub = df[df['label'] == lbl]['wachttijd']
            if len(sub) > 0:
                fig.add_trace(go.Histogram(
                    x=sub, nbinsx=30, name=lbl,
                    marker_color=kleur, opacity=.75,
                ))
        fig.add_vline(x=m['gem_wacht'], line_dash='dash', line_color='white',
                      annotation_text=f"Gem: {m['gem_wacht']:.0f} min",
                      annotation_font_color='white', annotation_position='top right')
        fig.add_vline(x=m['p90_wacht'], line_dash='dot', line_color='#f87171',
                      annotation_text=f"P90: {m['p90_wacht']:.0f} min",
                      annotation_font_color='#f87171', annotation_position='top right')
        fig.update_layout(
            title='Verdeling wachttijden per patiënttype',
            barmode='overlay',
            xaxis=dict(title='Wachttijd (minuten)', **G),
            yaxis=dict(title='Aantal patiënten', **G),
            height=320, legend=dict(orientation='h', y=-.28), **T,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        uren_sim = sorted(df['uur'].unique())
        fig2 = make_subplots(specs=[[{'secondary_y': True}]])
        fig2.add_trace(go.Bar(
            x=[f"{u}u" for u in uren_sim],
            y=[m['volume_per_uur'].get(u,0) for u in uren_sim],
            name='Aantal geholpen patiënten', marker_color='#1d4ed8', opacity=.45,
        ), secondary_y=True)
        fig2.add_trace(go.Scatter(
            x=[f"{u}u" for u in uren_sim],
            y=[m['per_uur'].get(u,0) for u in uren_sim],
            mode='lines+markers', name='Gem. wachttijd',
            line=dict(color='#f59e0b', width=2.5), marker=dict(size=6),
        ), secondary_y=False)
        fig2.update_layout(
            title='Gem. wachttijd per aankomstuur',
            height=320, legend=dict(orientation='h', y=-.28), **T,
            xaxis=dict(title='Aankomstuur van de patiënt (7u = 7:00–8:00)', **G),
        )
        fig2.update_yaxes(title_text='Gem. wachttijd (min)', secondary_y=False, **G)
        fig2.update_yaxes(title_text='Aantal geholpen patiënten in dat uur', secondary_y=True, showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        pt    = m['per_label']
        types = list(pt['mean'].keys())
        means = [pt['mean'][t] for t in types]
        cnts  = [pt['count'][t] for t in types]
        fig3  = go.Figure(go.Bar(
            x=types, y=means,
            marker_color=[kleuren.get(t,'#6b7280') for t in types],
            text=[f"{v:.1f} min<br>(n={c})" for v,c in zip(means,cnts)],
            textposition='auto',
        ))
        fig3.update_layout(
            title='Gem. wachttijd per patiënttype',
            xaxis=dict(title='Patiënttype', **G),
            yaxis=dict(title='Gem. wachttijd (min)', **G),
            height=300, showlegend=False, **T,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        cat_df = pd.DataFrame([
            {'Categorie': k.replace('cat_',''), 'Wachttijd': v}
            for k,v in m['per_categorie'].items()
        ]).sort_values('Wachttijd')
        fig4 = go.Figure(go.Bar(
            x=cat_df['Wachttijd'], y=cat_df['Categorie'],
            orientation='h', marker_color='#a78bfa',
            text=[f"{v:.1f} min" for v in cat_df['Wachttijd']],
            textposition='auto',
        ))
        fig4.update_layout(
            title='Gem. wachttijd per onderzoekscategorie',
            xaxis=dict(title='Gem. wachttijd (min)', **G),
            yaxis=dict(**G), height=300, showlegend=False, **T,
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: Zalen & Bezetting ──────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        qdf = m['wachtrij_df']
        cap = m['config']['wachtkamer_cap']
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=[uur_label(k) for k in qdf['klok_uur']],
            y=qdf['wachtkamer'],
            fill='tozeroy', fillcolor='rgba(59,130,246,.1)',
            line=dict(color='#3b82f6', width=1.5), name='Patiënten in wachtkamer',
        ))
        fig5.add_trace(go.Scatter(
            x=[uur_label(k) for k in qdf['klok_uur']],
            y=qdf['zalen_bezig'],
            line=dict(color='#f59e0b', width=1.5), name='Zalen in gebruik',
        ))
        fig5.add_hline(y=cap, line_dash='dash', line_color='#ef4444',
                       annotation_text=f'Max wachtkamer ({cap})',
                       annotation_font_color='#ef4444')
        fig5.update_layout(
            title='Wachtkamerbezetting & zaalgebruik doorheen de dag',
            xaxis=dict(title='Tijdstip van de dag', **G),
            yaxis=dict(title='Aantal', **G),
            height=300, legend=dict(orientation='h', y=-.28), **T,
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        zaal_gebruik = m.get('zaal_gebruik', {})
        zaal_wacht   = m.get('per_zaal', {})
        if zaal_gebruik:
            zalen_s = sorted(zaal_gebruik, key=lambda z: zaal_gebruik[z], reverse=True)
            fig6    = make_subplots(specs=[[{'secondary_y': True}]])
            fig6.add_trace(go.Bar(
                x=zalen_s, y=[zaal_gebruik.get(z,0) for z in zalen_s],
                name='Patiënten geholpen',
                marker_color=[ZAAL_KLEUR.get(z,'#6b7280') for z in zalen_s], opacity=.8,
            ), secondary_y=False)
            fig6.add_trace(go.Scatter(
                x=zalen_s, y=[zaal_wacht.get(z,0) for z in zalen_s],
                mode='markers+text', name='Gem. wachttijd',
                text=[f"{zaal_wacht.get(z,0):.0f}m" for z in zalen_s],
                textposition='top center',
                marker=dict(size=10, color='#f9fafb', symbol='diamond'),
            ), secondary_y=True)
            fig6.update_layout(
                title='Gebruik & wachttijd per zaal',
                height=300, legend=dict(orientation='h', y=-.28), **T,
                xaxis=dict(title='Zaal', **G),
            )
            fig6.update_yaxes(title_text='Patiënten geholpen', secondary_y=False, **G)
            fig6.update_yaxes(title_text='Gem. wachttijd (min)', secondary_y=True, showgrid=False)
            st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="hdr">Geschiktheidsmatrix (empirisch, 2021-2025)</div>',
                unsafe_allow_html=True)
    st.caption("Gebaseerd op werkelijk gebruik per zaal per onderzoekstype. Groen = geschikt (aanname A3).")
    cats = ['cat_thorax','cat_extremiteiten','cat_tafelwerk','cat_long_length','cat_overig']
    matrix = [{'Zaal': z, **{c.replace('cat_',''):
               '✅' if GESCHIKTHEID.get(z,{}).get(c,False) else '❌'
               for c in cats}}
              for z in m['config']['actieve_zalen']]
    st.dataframe(pd.DataFrame(matrix).set_index('Zaal'), use_container_width=True)

# ── TAB 3: Tijdsloten-analyse ─────────────────────────────────────────────────
with tab3:
    slot_pct_huidig = m.get('slot_pct', 0)
    actieve_slots   = m['config'].get('tijdsloten', [])

    st.markdown(f"""
    <div class="info-box">
    <b>Tijdsloten-logica (aanname A7)</b><br>
    Van de <b>~15 ambulante patiënten/dag zonder andere afspraak</b> wordt
    <b>{slot_pct_huidig}%</b> doorverwezen naar de gekozen tijdsloten
    ({len(actieve_slots)} halfuur-slots actief).
    De andere 85.4% ambulante patiënten (met consultatie) en alle H/D-patiënten
    zijn niet beïnvloed. De totale dagvraag blijft gelijk.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        uren_plot = list(range(SIM_START_UUR, SIM_EINDE_UUR))
        fig7 = go.Figure()
        # Baseline stuurbaar
        fig7.add_trace(go.Scatter(
            x=uren_plot, y=[AANKOMST_AMBULANT_STUURBAAR.get(h,0) for h in uren_plot],
            name='Stuurbaar — baseline', mode='lines+markers',
            line=dict(color='#6b7280', width=1.5, dash='dot'), marker=dict(size=4),
        ))
        # Scenario stuurbaar
        totaal_scenario = m['ambulant_profiel']
        fig7.add_trace(go.Scatter(
            x=uren_plot, y=[totaal_scenario.get(h,0) for h in uren_plot],
            name=f'Stuurbaar — scenario ({slot_pct_huidig}%)',
            mode='lines+markers',
            line=dict(color='#f59e0b', width=2.5), marker=dict(size=6),
        ))
        # Markeer actieve tijdsloten
        slot_uren_uniek = sorted(set(int(h) for h in actieve_slots))
        for h in slot_uren_uniek:
            fig7.add_vrect(x0=h-.5, x1=h+.5,
                           fillcolor='rgba(251,191,36,.1)', line_width=0)
        fig7.update_layout(
            title='Stuurbaar ambulant profiel — baseline vs scenario',
            xaxis=dict(title='Uur van de dag', dtick=1, **G),
            yaxis=dict(title='Gem. patiënten/uur', **G),
            height=300, legend=dict(orientation='h', y=-.28), **T,
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        uren_sim = sorted(df['uur'].unique())
        vol_lbl  = m.get('volume_label_uur', {})
        fig8 = go.Figure()
        for lbl, kleur in kleuren.items():
            vals = [vol_lbl.get(lbl, {}).get(u, 0) for u in uren_sim]
            if sum(vals) > 0:
                fig8.add_trace(go.Bar(
                    x=[f"{u}u" for u in uren_sim], y=vals,
                    name=lbl, marker_color=kleur, opacity=.85,
                ))
        fig8.update_layout(
            barmode='stack', title='Gesimuleerd volume per aankomstuur per type',
            xaxis=dict(title='Aankomstuur', **G),
            yaxis=dict(title='Aantal geholpen patiënten', **G),
            height=300, legend=dict(orientation='h', y=-.28), **T,
        )
        st.plotly_chart(fig8, use_container_width=True)

    # Actieve tijdsloten tonen
    if actieve_slots:
        st.markdown('<div class="hdr">Actieve tijdsloten</div>', unsafe_allow_html=True)
        slot_labels = [uur_label(s) for s in sorted(actieve_slots)]
        st.write("Patiënten zonder afspraak worden doorverwezen naar: " +
                 ", ".join(slot_labels))
    else:
        st.info("Geen tijdsloten actief. Selecteer slots in de sidebar om het effect te simuleren.")

    st.metric("Werkdruk CV",
              f"{m['werkdruk_cv']:.1f}%",
              help="Coëfficiënt van variatie van het volume per uur. "
                   "Lager = vlakkere werkdruk. Vergelijk scenario's om het effect van tijdsloten te zien.")

# ── TAB 4: Scenario-vergelijking ──────────────────────────────────────────────
with tab4:
    if len(st.session_state.scenarios) < 2:
        st.info("Draai minstens 2 scenario's om te vergelijken.\n\n"
                "**Tip:** Sla eerst een baseline op (0% tijdsloten, normaal drukte-niveau), "
                "dan varieer je één parameter per keer.")
    else:
        comp = []
        for naam, s in st.session_state.scenarios.items():
            if 'fout' in s: continue
            cfg = s['config']
            comp.append({
                'Scenario':          naam,
                'Drukte':            s.get('drukte','–').split(' ')[0],
                'Zalen':             len(cfg['actieve_zalen']),
                'Technici':          cfg['n_technici'],
                'Tijdsloten':        f"{s.get('slot_pct',0)}%",
                'Gem. wacht (min)':  round(s['gem_wacht'],1),
                'P90 wacht (min)':   round(s['p90_wacht'],1),
                'Balk-rate (%)':     round(s['balk_rate'],1),
                'Werkdruk CV (%)':   round(s['werkdruk_cv'],1),
                'Geholpen':          s['afgehandeld'],
            })
        comp_df = pd.DataFrame(comp)

        fig9 = go.Figure()
        for metric, kleur in [
            ('Gem. wacht (min)', '#10b981'),
            ('P90 wacht (min)',  '#f59e0b'),
            ('Werkdruk CV (%)',  '#60a5fa'),
            ('Balk-rate (%)',    '#ef4444'),
        ]:
            fig9.add_trace(go.Bar(
                name=metric, x=comp_df['Scenario'], y=comp_df[metric],
                marker_color=kleur,
            ))
        fig9.update_layout(
            barmode='group', title='Scenario-vergelijking (lager = beter)',
            xaxis=dict(**G), yaxis=dict(title='Waarde', **G),
            height=320, legend=dict(orientation='h', y=-.22), **T,
        )
        st.plotly_chart(fig9, use_container_width=True)

        # Ambulant profielen per scenario
        uren_plot = list(range(SIM_START_UUR, SIM_EINDE_UUR))
        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(
            x=uren_plot, y=[AANKOMST_AMBULANT_STUURBAAR.get(h,0) for h in uren_plot],
            name='Stuurbaar baseline', mode='lines',
            line=dict(color='#6b7280', width=1.5, dash='dot'),
        ))
        sc_kl = ['#60a5fa','#f59e0b','#34d399','#f87171','#a78bfa']
        for i, (naam, s) in enumerate(st.session_state.scenarios.items()):
            if 'fout' in s: continue
            fig10.add_trace(go.Scatter(
                x=uren_plot,
                y=[s['ambulant_profiel'].get(h,0) for h in uren_plot],
                name=f"{naam} ({s.get('slot_pct',0)}% slots)",
                mode='lines+markers',
                line=dict(color=sc_kl[i % len(sc_kl)], width=2),
                marker=dict(size=5),
            ))
        fig10.update_layout(
            title='Stuurbaar ambulant profiel per scenario',
            xaxis=dict(title='Uur van de dag', dtick=1, **G),
            yaxis=dict(title='Patiënten/uur (stuurbaar)', **G),
            height=260, legend=dict(orientation='h', y=-.3), **T,
        )
        st.plotly_chart(fig10, use_container_width=True)

        # --- NIEUWE GRAFIEK: Totaal geholpen volume per uur ---
        st.markdown('<div class="hdr">Totaal geholpen patiënten per uur (Alle types)</div>', unsafe_allow_html=True)
        fig11 = go.Figure()
        
        for i, (naam, s) in enumerate(st.session_state.scenarios.items()):
            if 'fout' in s: continue
            
            # Haal het totale volume per uur op uit de simulatie metrics
            vol_dict = s.get('volume_per_uur', {})
            uren_beschikbaar = sorted(vol_dict.keys())
            
            fig11.add_trace(go.Scatter(
                x=[f"{u}u" for u in uren_beschikbaar],
                y=[vol_dict[u] for u in uren_beschikbaar],
                name=f"{naam}",
                mode='lines+markers',
                line=dict(color=sc_kl[i % len(sc_kl)], width=2),
                marker=dict(size=6)
            ))
            
        fig11.update_layout(
            title='Totale ziekenhuiswerkdruk per scenario (afvlakking van de piek)',
            xaxis=dict(title='Aankomstuur van de patiënt', **G),
            yaxis=dict(title='Totaal geholpen patiënten', **G),
            height=280, legend=dict(orientation='h', y=-.3), **T,
        )
        st.plotly_chart(fig11, use_container_width=True)
        # --------------------------------------------------------

        st.dataframe(
            comp_df.style.highlight_min(
                subset=['Gem. wacht (min)','P90 wacht (min)','Balk-rate (%)','Werkdruk CV (%)'],
                color='#064e3b',
            ),
            use_container_width=True, hide_index=True,
        )

        if st.button("🗑 Wis alle scenario's"):
            st.session_state.scenarios = {}
            st.session_state.laatste   = None
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='font-size:.7rem;color:#6b7280;text-align:center;line-height:2'>
  SimPy discrete-event simulatie · Plotly · Streamlit · KWS-data 2021-2025<br>
  Simulatievenster: {SIM_START_UUR}u–{SIM_EINDE_UUR}u · Stuurbaar ambulant: ~15/dag (14.6%) ·
  Niet-stuurbaar ambulant: ~87/dag · Gehospitaliseerd: ~23/dag · Dagziekenhuis: ~6/dag
</div>
""", unsafe_allow_html=True)
