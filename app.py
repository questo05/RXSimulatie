"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DASHBOARD — RX Wachtlijstsimulatie UZ Gasthuisberg                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Starten: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulatie_nieuw import (
    RXSimulatie,
    AANKOMST_AMBULANT, AANKOMST_HOSP, AANKOMST_DAG,
    GESCHIKTHEID, DEFAULT_ZALEN, DEFAULT_TECHNICI, DEFAULT_WACHTKAMER,
    bouw_ambulant_profiel,
)

# ── Pagina-instellingen ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RX Simulatie — UZ Gasthuisberg",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stijl ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.kpi { background:#111827; border:1px solid #1f2937; border-radius:10px;
       padding:1rem 1.2rem; margin-bottom:0.4rem; }
.kpi-lbl { font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
           color:#6b7280; text-transform:uppercase; letter-spacing:.09em; margin-bottom:.2rem; }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:1.7rem;
           font-weight:600; color:#f9fafb; line-height:1; }
.kpi-unit { font-size:.78rem; color:#6b7280; margin-left:.2rem; }
.kpi-ok   { color:#34d399; font-size:.72rem; margin-top:.15rem; }
.kpi-warn { color:#fbbf24; font-size:.72rem; margin-top:.15rem; }
.kpi-bad  { color:#f87171; font-size:.72rem; margin-top:.15rem; }

.hdr { font-family:'IBM Plex Mono',monospace; font-size:.68rem; color:#6b7280;
       text-transform:uppercase; letter-spacing:.1em; border-bottom:1px solid #1f2937;
       padding-bottom:.3rem; margin:1.3rem 0 .8rem 0; }

.assumptie { background:#1a2035; border-left:3px solid #3b82f6;
             padding:.6rem .9rem; border-radius:0 6px 6px 0;
             font-size:.78rem; color:#93c5fd; margin-bottom:.4rem; }

.stButton>button { background:#059669; color:white; border:none; border-radius:6px;
                   font-family:'IBM Plex Mono',monospace; font-size:.85rem;
                   padding:.5rem 1rem; width:100%; }
.stButton>button:hover { background:#10b981; }
</style>
""", unsafe_allow_html=True)

T = dict(paper_bgcolor="#111827", plot_bgcolor="#111827",
         font=dict(color="#d1d5db", family="IBM Plex Mono"))
G = dict(gridcolor="#1f2937", zerolinecolor="#1f2937")
KLEUREN = {'Ambulant': '#60a5fa', 'Gehospitaliseerd': '#f87171', 'Dagziekenhuis': '#34d399'}
ZAAL_KLEUR = {'G5':'#f59e0b','G15':'#a78bfa','G3':'#34d399','G4':'#60a5fa','G2':'#f87171','G1':'#6b7280'}

def kpi(lbl, val, unit='', sig=None):
    s = f'<div class="kpi-{sig[0]}">{sig[1]}</div>' if sig else ''
    return f'<div class="kpi"><div class="kpi-lbl">{lbl}</div><div class="kpi-val">{val}<span class="kpi-unit">{unit}</span></div>{s}</div>'

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 RX Simulatie")
    st.markdown("*UZ Gasthuisberg — Radiologie*")
    st.divider()

    # ── Infrastructuur ────────────────────────────────────────────────────────
    st.markdown("### 📦 Infrastructuur")
    st.caption("Welke zalen en technici zijn beschikbaar?")

    alle_zalen = ['G5', 'G15', 'G3', 'G4', 'G2', 'G1']
    actieve_zalen = st.multiselect(
        "Actieve RX-zalen",
        options=alle_zalen,
        default=DEFAULT_ZALEN,
        help="G5=thorax, G15=full-length, G2/G3/G4=gemengd, G1=backup"
    )
    if not actieve_zalen:
        actieve_zalen = DEFAULT_ZALEN

    n_technici = st.slider("Radiotechnici", 1, 12, DEFAULT_TECHNICI,
                           help="Aantal beschikbare technici tegelijk (aanname: 1 per onderzoek)")
    wk_cap = st.slider("Wachtkamercapaciteit", 5, 80, DEFAULT_WACHTKAMER,
                       help="Patiënten die wachtkamer niet in kunnen vertrekken direct (balking)")

    # ── Wachtrij-logica ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ Wachtrij-logica")
    fifo = st.toggle("FIFO-wachtrij", value=True,
                     help="AAN = first-in-first-out (geen prioriteit). UIT = gehospitaliseerden (H) krijgen lichte voorrang.")
    balk_drempel = st.slider("Balking drempel (min)", 15, 180, 60,
                             help="Patiënt verlaat wachtkamer als wachttijd > X minuten")

    # ── Simulatieduur ─────────────────────────────────────────────────────────
    st.markdown("### 🕐 Simulatie")
    sim_uren   = st.slider("Simulatieduur (uren)", 4, 24, 12)
    schaal_A   = st.slider("Volume ambulant (×)", 0.5, 2.0, 1.0, 0.05,
                           help="1.0 = gemiddelde werkdag 2021-2025 (~102 ambulante patiënten)")
    schaal_H   = st.slider("Volume gehospitaliseerd (×)", 0.5, 2.0, 1.0, 0.05)

    st.divider()

    # ── Tijdsloten ambulant ───────────────────────────────────────────────────
    st.markdown("### 🎯 Tijdsloten ambulant")
    st.markdown("""
    <div style='font-size:.72rem;color:#6b7280;line-height:1.6;margin-bottom:.5rem'>
    Stuur ambulante walk-ins naar specifieke uren om de werkdruk te spreiden.
    <br><br>
    <b>Hoe het werkt:</b> een percentage van de ambulante patiënten die normaal
    in piekuren toekomen wordt herverdeeld naar de gekozen tijdsloten.
    </div>
    """, unsafe_allow_html=True)

    slot_pct = st.slider("% ambulanten naar tijdsloten", 0, 100, 0, 5,
                         help="0% = huidige situatie zonder sturing")
    slot_fractie = slot_pct / 100

    tijdsloten = st.multiselect(
        "Beschikbare tijdsloten (uren)",
        options=list(range(6, 18)),
        default=[12, 13, 16, 17],
        help="Uren waarop ambulante patiënten worden ingepland"
    )

    st.divider()
    scenario_naam = st.text_input("Scenarionaam", "Baseline")
    seed          = st.number_input("Random seed", 0, 9999, 42)
    run_btn       = st.button("▶ Start simulatie", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}
if 'laatste' not in st.session_state:
    st.session_state.laatste = None

# ── Simulatie starten ─────────────────────────────────────────────────────────
if run_btn:
    if not actieve_zalen:
        st.error("Selecteer minstens één zaal.")
    else:
        config = {
            'actieve_zalen':   actieve_zalen,
            'n_technici':      n_technici,
            'wachtkamer_cap':  wk_cap,
            'balk_drempel_min': balk_drempel,
            'sim_uren':        sim_uren,
            'schaal_ambulant': schaal_A,
            'schaal_hosp':     schaal_H,
            'tijdsloten':      tijdsloten if slot_pct > 0 else [],
            'slot_fractie':    slot_fractie,
            'fifo':            fifo,
        }
        with st.spinner(f"Simulatie '{scenario_naam}' bezig..."):
            res = RXSimulatie(config=config, seed=seed).run()
            res['naam']     = scenario_naam
            res['slot_pct'] = slot_pct
            st.session_state.scenarios[scenario_naam] = res
            st.session_state.laatste = scenario_naam
        if 'fout' not in res:
            st.success(f"✓ Klaar — {res['afgehandeld']} patiënten afgehandeld, {res['gebalkt']} gebalkt")

# ══════════════════════════════════════════════════════════════════════════════
# HOOFDPAGINA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-family:IBM Plex Mono;font-size:1.4rem;color:#f9fafb;margin-bottom:0'>
  RX Wachtlijstsimulatie — UZ Gasthuisberg
</h1>
<p style='color:#6b7280;font-size:.85rem;margin-top:.3rem'>
  Radiologie walk-in · SimPy discrete-event simulatie · KWS-data 2021-2025
</p>
""", unsafe_allow_html=True)

# Assumpties tonen als er nog geen simulatie is
if not st.session_state.scenarios:
    st.info("👈 Stel parameters in en klik op **Start simulatie**.")

    st.markdown('<div class="hdr">Modelassumpties</div>', unsafe_allow_html=True)
    assumpties = [
        ("A1 — FIFO", "Geen prioriteit tussen patiënten (first-in-first-out). Optioneel: gehospitaliseerden (H) krijgen lichte voorrang omdat een verpleegkundige op hen wacht."),
        ("A2 — Resources", "Elke patiënt heeft 1 zaal + 1 technicus nodig. Geen multi-zaal beheer per technicus."),
        ("A3 — Geschiktheid", "G5 = thorax, G15 = full-length, G2/G3/G4 = gemengd, G1 = backup. Patiënt wacht op geschikte zaal — gaat niet naar ongeschikte."),
        ("A4 — Exam duraties", "Geschat via tijdsverschil opeenvolgende acta per zaal (mediaan 7-11 min). Echte begin/eindtijden ontbreken in de data."),
        ("A5 — Balking", "Patiënt verlaat wachtkamer na X minuten wachten (configureerbaar). Wachtkamer vol = directe afwijzing."),
        ("A6 — Aankomst", "Poisson-proces per uur op basis van gemiddeld aankomstprofiel werkdagen 2021-2025."),
        ("A7 — Tijdsloten", "Ambulante patiënten worden herdistribueerd naar gekozen uren. Buiten die uren: proportioneel minder aankomsten."),
    ]
    for titel, tekst in assumpties:
        st.markdown(f'<div class="assumptie"><b>{titel}</b><br>{tekst}</div>', unsafe_allow_html=True)

    # Aankomstprofiel tonen als preview
    st.markdown('<div class="hdr">Empirisch aankomstprofiel (werkdag 2021-2025)</div>', unsafe_allow_html=True)
    uren = list(range(5, 18))
    fig = go.Figure()
    for profiel, naam, kleur in [
        (AANKOMST_AMBULANT, 'Ambulant (A) — stuurbaar', '#60a5fa'),
        (AANKOMST_HOSP,     'Gehospitaliseerd (H)',      '#f87171'),
        (AANKOMST_DAG,      'Dagziekenhuis (D)',          '#34d399'),
    ]:
        fig.add_trace(go.Bar(
            x=uren, y=[profiel.get(h, 0) for h in uren],
            name=naam, marker_color=kleur, opacity=.85,
        ))
    fig.update_layout(
        barmode='stack', height=320,
        xaxis=dict(title='Uur van de dag', dtick=1, **G),
        yaxis=dict(title='Patiënten/uur (gem. werkdag)', **G),
        legend=dict(orientation='h', y=-.22), **T,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ── Resultaten ophalen ────────────────────────────────────────────────────────
m = st.session_state.scenarios[st.session_state.laatste]
if 'fout' in m:
    st.error(m['fout'])
    st.stop()

df = m['patient_df']

# ══════════════════════════════════════════════════════════════════════════════
# KPI-KAARTEN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hdr">Kernresultaten</div>', unsafe_allow_html=True)

c = st.columns(7)
with c[0]: st.markdown(kpi('Aankomsten',    m['totaal']),             unsafe_allow_html=True)
with c[1]: st.markdown(kpi('Afgehandeld',   m['afgehandeld']),        unsafe_allow_html=True)

w = m['gem_wacht']
sig = ('ok','✓ OK') if w<20 else ('warn','⚠ Verhoogd') if w<40 else ('bad','✗ Hoog')
with c[2]: st.markdown(kpi('Gem. wacht', f"{w:.1f}", 'min', sig),     unsafe_allow_html=True)
with c[3]: st.markdown(kpi('Mediaan wacht', f"{m['mediaan_wacht']:.1f}", 'min'), unsafe_allow_html=True)
with c[4]: st.markdown(kpi('P90 wacht',  f"{m['p90_wacht']:.1f}", 'min'),        unsafe_allow_html=True)

br = m['balk_rate']
sig = ('ok','✓ Laag') if br<3 else ('warn','⚠ Matig') if br<8 else ('bad','✗ Hoog')
with c[5]: st.markdown(kpi('Balk-rate', f"{br:.1f}", '%', sig),       unsafe_allow_html=True)

cv = m['werkdruk_cv']
sig = ('ok','✓ Vlak') if cv<40 else ('warn','⚠ Piekerig') if cv<60 else ('bad','✗ Hoge piek')
with c[6]: st.markdown(kpi('Werkdruk CV', f"{cv:.1f}", '%', sig),     unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB-STRUCTUUR
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Wachttijden",
    "🏥 Zalen & Bezetting",
    "🎯 Tijdsloten-analyse",
    "🔀 Scenario-vergelijking",
])

# ── TAB 1: Wachttijden ────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # Wachttijdverdeling per type
        fig = go.Figure()
        for t, kleur in KLEUREN.items():
            sub = df[df['type_label'] == t]['wachttijd']
            if len(sub):
                fig.add_trace(go.Histogram(
                    x=sub, nbinsx=35, name=t,
                    marker_color=kleur, opacity=.75,
                ))
        fig.add_vline(x=m['gem_wacht'], line_dash='dash', line_color='#60a5fa',
                      annotation_text=f"Gem {m['gem_wacht']:.0f}m",
                      annotation_font_color='#60a5fa')
        fig.add_vline(x=m['p90_wacht'], line_dash='dot', line_color='#f87171',
                      annotation_text=f"P90 {m['p90_wacht']:.0f}m",
                      annotation_font_color='#f87171')
        fig.update_layout(
            title='Wachttijdverdeling per patiënttype', barmode='overlay',
            xaxis=dict(title='Wachttijd (min)', **G),
            yaxis=dict(title='Aantal patiënten', **G),
            height=320, legend=dict(orientation='h', y=-.22), **T,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Wachttijd + volume per uur
        uren_sim = sorted(df['uur'].unique())
        fig2 = make_subplots(specs=[[{'secondary_y': True}]])
        fig2.add_trace(go.Bar(
            x=uren_sim, y=[m['volume_per_uur'].get(h,0) for h in uren_sim],
            name='Volume', marker_color='#1d4ed8', opacity=.4,
        ), secondary_y=True)
        fig2.add_trace(go.Scatter(
            x=uren_sim, y=[m['per_uur'].get(h,0) for h in uren_sim],
            mode='lines+markers', name='Gem. wachttijd',
            line=dict(color='#f59e0b', width=2.5), marker=dict(size=6),
        ), secondary_y=False)
        fig2.update_layout(
            title='Wachttijd & volume per uur', height=320,
            xaxis=dict(title='Uur', dtick=1, **G),
            legend=dict(orientation='h', y=-.22), **T,
        )
        fig2.update_yaxes(title_text='Wachttijd (min)', secondary_y=False, **G)
        fig2.update_yaxes(title_text='Patiënten', secondary_y=True, showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Wachttijd per patiënttype
        pt     = m['per_type']
        types  = list(pt['mean'].keys())
        means  = [pt['mean'][t] for t in types]
        counts = [pt['count'][t] for t in types]
        fig3 = go.Figure(go.Bar(
            x=types, y=means,
            marker_color=[KLEUREN.get(t,'#6b7280') for t in types],
            text=[f"{v:.1f}m (n={c})" for v,c in zip(means,counts)],
            textposition='auto',
        ))
        fig3.update_layout(
            title='Gem. wachttijd per patiënttype',
            xaxis=dict(**G), yaxis=dict(title='Wachttijd (min)', **G),
            height=280, showlegend=False, **T,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Wachttijd per onderzoekscategorie
        cat_df = pd.DataFrame([
            {'Categorie': k, 'Wachttijd': v}
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
            xaxis=dict(title='Wachttijd (min)', **G),
            yaxis=dict(**G), height=280, showlegend=False, **T,
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: Zalen & Bezetting ──────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        # Wachtkamerbezetting in de tijd
        qdf = m['wachtrij_df']
        cap = m['config']['wachtkamer_cap']
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=qdf['tijd']/60, y=qdf['wachtkamer'],
            fill='tozeroy', fillcolor='rgba(59,130,246,.1)',
            line=dict(color='#3b82f6', width=1.5), name='Wachtkamer',
        ))
        fig5.add_trace(go.Scatter(
            x=qdf['tijd']/60, y=qdf['zalen_bezig'],
            line=dict(color='#f59e0b', width=1.5), name='Zalen in gebruik',
        ))
        fig5.add_hline(y=cap, line_dash='dash', line_color='#ef4444',
                       annotation_text=f'Cap ({cap})', annotation_font_color='#ef4444')
        fig5.update_layout(
            title='Wachtkamer & zaalgebruik in de tijd',
            xaxis=dict(title='Simulatietijd (uur)', **G),
            yaxis=dict(title='Aantal', **G),
            height=300, legend=dict(orientation='h', y=-.22), **T,
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        # Zaalgebruik: aantal patiënten per zaal
        zaal_gebruik = m.get('zaal_gebruik', {})
        zaal_wacht   = m.get('per_zaal', {})
        if zaal_gebruik:
            zalen_sorted = sorted(zaal_gebruik, key=lambda z: zaal_gebruik[z], reverse=True)
            fig6 = make_subplots(specs=[[{'secondary_y': True}]])
            fig6.add_trace(go.Bar(
                x=zalen_sorted,
                y=[zaal_gebruik.get(z,0) for z in zalen_sorted],
                name='Patiënten',
                marker_color=[ZAAL_KLEUR.get(z,'#6b7280') for z in zalen_sorted],
                opacity=.8,
            ), secondary_y=False)
            fig6.add_trace(go.Scatter(
                x=zalen_sorted,
                y=[zaal_wacht.get(z,0) for z in zalen_sorted],
                mode='markers', name='Gem. wachttijd',
                marker=dict(size=10, color='#f9fafb', symbol='diamond'),
            ), secondary_y=True)
            fig6.update_layout(
                title='Gebruik per zaal + gem. wachttijd',
                height=300, legend=dict(orientation='h', y=-.22), **T,
                xaxis=dict(**G),
            )
            fig6.update_yaxes(title_text='Patiënten', secondary_y=False, **G)
            fig6.update_yaxes(title_text='Gem. wachttijd (min)', secondary_y=True, showgrid=False)
            st.plotly_chart(fig6, use_container_width=True)

    # Geschiktheidsmatrix tonen
    st.markdown('<div class="hdr">Geschiktheidsmatrix zalen/onderzoeken (empirisch, 2021-2025)</div>',
                unsafe_allow_html=True)
    st.caption("Groen = geschikt op basis van empirisch gebruik. Gebaseerd op aanname A3.")

    actieve = m['config']['actieve_zalen']
    cats    = ['cat_thorax','cat_extremiteiten','cat_tafelwerk','cat_long_length','cat_overig']
    matrix_data = []
    for zaal in actieve:
        rij = {'Zaal': zaal}
        for cat in cats:
            rij[cat.replace('cat_','')] = '✅' if GESCHIKTHEID.get(zaal,{}).get(cat,False) else '❌'
        matrix_data.append(rij)
    st.dataframe(pd.DataFrame(matrix_data).set_index('Zaal'), use_container_width=True)

# ── TAB 3: Tijdsloten-analyse ─────────────────────────────────────────────────
with tab3:
    slot_pct_huidig = m.get('slot_pct', 0)

    st.markdown(f"""
    <div class="assumptie">
    <b>Tijdsloten-logica (aanname A7)</b><br>
    {slot_pct_huidig}% van de ambulante patiënten werd herdistribueerd naar de gekozen tijdsloten.
    Piekuren krijgen proportioneel minder aankomsten; tijdsloten proportioneel meer.
    De totale dagvraag blijft gelijk.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Ambulant profiel: baseline vs scenario
        uren_plot = list(range(5, 18))
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=uren_plot, y=[AANKOMST_AMBULANT.get(h,0) for h in uren_plot],
            name='Baseline (geen sturing)', mode='lines+markers',
            line=dict(color='#6b7280', width=1.5, dash='dot'), marker=dict(size=4),
        ))
        fig7.add_trace(go.Scatter(
            x=uren_plot, y=[m['ambulant_profiel'].get(h,0) for h in uren_plot],
            name=f'Scenario ({slot_pct_huidig}% tijdsloten)',
            mode='lines+markers',
            line=dict(color='#f59e0b', width=2.5), marker=dict(size=6),
        ))
        # Markeer tijdsloten
        for h in m['config'].get('tijdsloten', []):
            fig7.add_vrect(x0=h-.5, x1=h+.5,
                           fillcolor='rgba(251,191,36,.08)', line_width=0)
        fig7.update_layout(
            title='Ambulant aankomstprofiel — baseline vs scenario',
            xaxis=dict(title='Uur', dtick=1, **G),
            yaxis=dict(title='Patiënten/uur', **G),
            height=300, legend=dict(orientation='h', y=-.22), **T,
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        # Gestapeld volume per uur per type
        uren_sim = sorted(df['uur'].unique())
        vol_type = m.get('volume_type_uur', {})
        fig8 = go.Figure()
        for t, kleur in KLEUREN.items():
            vals = [vol_type.get(t, {}).get(h, 0) for h in uren_sim]
            fig8.add_trace(go.Bar(x=uren_sim, y=vals, name=t,
                                  marker_color=kleur, opacity=.85))
        fig8.update_layout(
            barmode='stack', title='Gesimuleerd volume per uur per type',
            xaxis=dict(title='Uur', dtick=1, **G),
            yaxis=dict(title='Patiënten', **G),
            height=300, legend=dict(orientation='h', y=-.22), **T,
        )
        st.plotly_chart(fig8, use_container_width=True)

    # Werkdruk CV uitleg
    st.metric(
        "Werkdruk CV (coëfficiënt van variatie)",
        f"{m['werkdruk_cv']:.1f}%",
        help="Lagere waarde = vlakkere werkdruk over de dag. "
             "Vergelijk scenario's om effect van tijdsloten te zien."
    )

# ── TAB 4: Scenario-vergelijking ──────────────────────────────────────────────
with tab4:
    if len(st.session_state.scenarios) < 2:
        st.info("Draai minstens 2 scenario's om te vergelijken. "
                "Tip: sla eerst een 'Baseline' op (0% tijdsloten), "
                "dan een scenario met bv. 30% tijdsloten.")
    else:
        comp = []
        for naam, s in st.session_state.scenarios.items():
            if 'fout' in s: continue
            cfg = s['config']
            comp.append({
                'Scenario':        naam,
                'Zalen':           len(cfg['actieve_zalen']),
                'Technici':        cfg['n_technici'],
                'Tijdsloten':      f"{s.get('slot_pct',0)}%",
                'Gem. wacht (min)': round(s['gem_wacht'], 1),
                'P90 wacht (min)':  round(s['p90_wacht'], 1),
                'Balk-rate (%)':    round(s['balk_rate'], 1),
                'Werkdruk CV (%)':  round(s['werkdruk_cv'], 1),
                'Afgehandeld':      s['afgehandeld'],
            })
        comp_df = pd.DataFrame(comp)

        # Staafdiagram vergelijking
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
        uren_plot = list(range(5, 18))
        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(
            x=uren_plot, y=[AANKOMST_AMBULANT.get(h,0) for h in uren_plot],
            name='Empirisch baseline', mode='lines',
            line=dict(color='#6b7280', width=1.5, dash='dot'),
        ))
        sc_kleuren = ['#60a5fa','#f59e0b','#34d399','#f87171','#a78bfa']
        for i, (naam, s) in enumerate(st.session_state.scenarios.items()):
            if 'fout' in s: continue
            fig10.add_trace(go.Scatter(
                x=uren_plot,
                y=[s['ambulant_profiel'].get(h,0) for h in uren_plot],
                name=f"{naam} ({s.get('slot_pct',0)}%)",
                mode='lines+markers',
                line=dict(color=sc_kleuren[i % len(sc_kleuren)], width=2),
                marker=dict(size=5),
            ))
        fig10.update_layout(
            title='Ambulant profiel per scenario',
            xaxis=dict(title='Uur', dtick=1, **G),
            yaxis=dict(title='Patiënten/uur', **G),
            height=280, legend=dict(orientation='h', y=-.3), **T,
        )
        st.plotly_chart(fig10, use_container_width=True)

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
st.markdown("""
<div style='font-size:.7rem;color:#6b7280;text-align:center;line-height:2'>
  SimPy discrete-event simulatie · Plotly · Streamlit<br>
  KWS-data 2021-2025 · 165.259 contacten · 337k acta<br>
  Patiënttypes: A Ambulant (78%) · H Gehospitaliseerd (18%) · D Dagziekenhuis (5%)<br>
  Zalen: G5=thorax · G15=full-length · G2/G3/G4=gemengd · G1=backup
</div>
""", unsafe_allow_html=True)
