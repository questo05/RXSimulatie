"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RX WACHTLIJSTSIMULATIE — UZ Gasthuisberg                                   ║
║  Discrete-event simulatie via SimPy                                          ║
║  Geparametriseerd met KWS-data 2021-2025 (165.259 contacten, 337k acta)     ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUCTUUR VAN DIT BESTAND
─────────────────────────
1. PARAMETERS        — alle empirische waarden uit de data, makkelijk aanpasbaar
2. HULPFUNCTIES      — sampling, categorie-toewijzing, geschiktheidscheck
3. PATIENT-DATAMODEL — wat we bijhouden per patiënt
4. SIMULATIEKLASSE   — de eigenlijke SimPy-simulatie
5. METRICS           — berekening van resultaten na de simulatie

ASSUMPTIES & KEUZES (te bespreken met het ziekenhuis)
──────────────────────────────────────────────────────
A1. FIFO-wachtrij: geen prioriteit tussen patiënten, first-in-first-out.
    Uitzondering: gehospitaliseerde patiënten (AD Type H) krijgen
    lichte voorrang omdat ze afhankelijk zijn van een verpleegkundige
    die wacht — dit is een aanname die kan worden uitgeschakeld.

A2. Elke patiënt heeft 1 zaal + 1 technicus nodig tegelijk.
    We modelleren geen situatie waar 1 technicus meerdere zalen beheert.

A3. Geschiktheidsmatrix: gebaseerd op empirisch gebruik 2021-2025.
    G5 = thorax, G15 = full-length, G2/G3/G4 = gemengd, G1 = backup.
    Als een geschikte zaal bezet is, wacht de patiënt — hij gaat niet
    naar een minder geschikte zaal (conservatieve aanname).

A4. Exam duraties: geschat via tijdsverschil tussen opeenvolgende acta
    op dezelfde zaal. Dit is een benadering — echte begin/eindtijden
    zijn niet beschikbaar in de data.

A5. Balking: patiënten die langer dan X minuten wachten verlaten de
    wachtkamer. Drempel is configureerbaar in het dashboard.

A6. Aankomstproces: Poisson-proces per uur (exponentieel verdeelde
    interaankomsttijden), gebaseerd op gemiddeld aankomstprofiel
    werkdagen 2021-2025.

A7. Tijdsloten ambulant: als tijdsloten actief zijn, worden ambulante
    patiënten herdistribueerd naar de gekozen uren. Buiten die uren
    komen geen ambulante patiënten toe.
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMETERS
#    Alle waarden hieronder zijn empirisch bepaald uit KWS-data 2021-2025.
#    Pas hier aan als het ziekenhuis andere waarden heeft.
# ══════════════════════════════════════════════════════════════════════════════

# ── Aankomstprofielen ─────────────────────────────────────────────────────────
# Gemiddeld aantal patiënten per uur per werkdag, per AD Type
# Bron: 165.259 contacten, weekdagen, 2021-2025 (1266 werkdagen)

AANKOMST_AMBULANT = {          # AD Type A — walk-in, stuurbaar
    6: 0.13, 7: 3.46, 8: 11.39, 9: 14.22, 10: 14.03,
    11: 9.50, 12: 7.03, 13: 10.84, 14: 13.31, 15: 10.52,
    16: 5.81, 17: 1.21,
}

AANKOMST_HOSP = {              # AD Type H — gehospitaliseerd, niet stuurbaar
    5: 0.07, 6: 11.83, 7: 0.52, 8: 1.49, 9: 1.33, 10: 1.57,
    11: 1.21, 12: 0.65, 13: 1.63, 14: 1.30, 15: 0.81,
    16: 0.52, 17: 0.13,
}

AANKOMST_DAG = {               # AD Type D — dagziekenhuis, niet stuurbaar
    7: 0.08, 8: 0.92, 9: 1.06, 10: 0.92, 11: 0.53,
    12: 0.44, 13: 0.76, 14: 0.59, 15: 0.35, 16: 0.18, 17: 0.04,
}

# ── Onderzoekscategorieën ─────────────────────────────────────────────────────
# Gebaseerd op categorisatie van Flore, toegepast op 337k acta

CATEGORIE_MIX = {
    'cat_thorax':       0.2727,
    'cat_extremiteiten':0.2919,   # bovenste + onderste extremiteiten samen
    'cat_tafelwerk':    0.1555,   # wervelkolom, bekken, heup, abdomen
    'cat_long_length':  0.0680,   # full-spine, EOS
    'cat_overig':       0.1119,
}

def categorie_van_omschrijving(omschrijving: str) -> str:
    """Zet omschrijving om naar onderzoekscategorie (logica van Flore)."""
    if pd.isna(omschrijving):
        return 'cat_overig'
    s = str(omschrijving).upper()
    if 'THORAX' in s or 'LONG' in s:
        return 'cat_thorax'
    if any(x in s for x in ['FULL', 'SPINE', 'LENGTE', 'EOS']):
        return 'cat_long_length'
    if any(x in s for x in ['WERVEL', 'NEK', 'RUG', 'BEKKEN', 'SACRUM',
                              'HEUP', 'HIP', 'ABDOMEN', 'BUIK']):
        return 'cat_tafelwerk'
    if any(x in s for x in ['HAND', 'POLS', 'ARM', 'ELBOW', 'VINGER',
                              'SCHOUDER', 'KNIE', 'VOET', 'ENKEL', 'BEEN', 'FEMUR']):
        return 'cat_extremiteiten'
    return 'cat_overig'

# ── Exam duraties per categorie ───────────────────────────────────────────────
# Geschat via tijdsverschil opeenvolgende acta per zaal (minuten)
# Bron: 160k+ acta met tijdstempel op G2/G3/G4/G5/G15
# Kolommen: median, std (voor lognormale fit)

EXAM_DURATIE = {
    'cat_thorax':        {'median': 7,  'std': 8.8},
    'cat_extremiteiten': {'median': 8,  'std': 9.3},
    'cat_tafelwerk':     {'median': 7,  'std': 9.4},
    'cat_long_length':   {'median': 11, 'std': 13.0},
    'cat_overig':        {'median': 7,  'std': 9.5},
}

# ── Geschiktheidsmatrix zalen/categorieën ─────────────────────────────────────
# Gebaseerd op empirisch gebruik 2021-2025 (zie % per zaal in analyse)
#
# Interpretatie:
#   G5  → bijna exclusief thorax (99.5% van G5-gebruik)
#   G15 → bijna exclusief full-length/EOS (95.3%)
#   G2, G3, G4 → gemengd: extremiteiten + tafelwerk + thorax
#   G1  → backup voor alles
#
# Een zaal is 'geschikt' als het type daar regelmatig gedaan wordt.
# Aanname A3: patiënt wacht op geschikte zaal, gaat niet naar ongeschikte.

GESCHIKTHEID = {
    # zaal          thorax  extremiteiten  tafelwerk  long_length  overig
    'G5':  {'cat_thorax': True,  'cat_extremiteiten': False, 'cat_tafelwerk': False, 'cat_long_length': False, 'cat_overig': False},
    'G15': {'cat_thorax': False, 'cat_extremiteiten': False, 'cat_tafelwerk': False, 'cat_long_length': True,  'cat_overig': False},
    'G3':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': False, 'cat_overig': True},
    'G4':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': False, 'cat_overig': True},
    'G2':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': False, 'cat_overig': True},
    'G1':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': True,  'cat_overig': True},
}

# Volgorde van voorkeur per categorie (meest geschikte zaal eerst)
ZAAL_VOORKEUR = {
    'cat_thorax':        ['G5', 'G3', 'G4', 'G2', 'G1'],
    'cat_long_length':   ['G15', 'G1'],
    'cat_extremiteiten': ['G3', 'G4', 'G2', 'G1'],
    'cat_tafelwerk':     ['G3', 'G4', 'G2', 'G1'],
    'cat_overig':        ['G3', 'G4', 'G2', 'G1'],
}

# Default infrastructuur (aanpasbaar in dashboard)
DEFAULT_ZALEN      = ['G5', 'G15', 'G3', 'G4', 'G2']   # G1 standaard niet actief
DEFAULT_TECHNICI   = 6
DEFAULT_WACHTKAMER = 30


# ══════════════════════════════════════════════════════════════════════════════
# 2. HULPFUNCTIES
# ══════════════════════════════════════════════════════════════════════════════

def sample_categorie() -> str:
    """Trek een willekeurige onderzoekscategorie op basis van empirische mix."""
    cats  = list(CATEGORIE_MIX.keys())
    probs = list(CATEGORIE_MIX.values())
    # Normaliseer voor het geval de som niet exact 1 is
    probs = np.array(probs) / sum(probs)
    return np.random.choice(cats, p=probs)


def sample_duratie(categorie: str) -> float:
    """
    Trek een exam-duratie uit een lognormale verdeling.
    Parameters afgeleid van empirische mediaan en std.
    Minimum is 2 minuten (praktische ondergrens).
    """
    p = EXAM_DURATIE[categorie]
    median, std = p['median'], p['std']
    # Lognormale parametrisatie via mediaan en std
    sigma2 = np.log(1 + (std / median) ** 2)
    mu     = np.log(median) - 0.5 * sigma2
    return max(2.0, np.random.lognormal(mu, np.sqrt(sigma2)))


def get_aankomstrate(profiel: dict, sim_minuten: float, schaal: float = 1.0) -> float:
    """Geeft de aankomstrate (patiënten/uur) voor het huidige simulatie-uur."""
    uur = int((sim_minuten / 60) % 24)
    return profiel.get(uur, 0.0) * schaal


def bouw_ambulant_profiel(tijdsloten: List[int], fractie: float) -> dict:
    """
    Herverdeelt ambulante aankomsten naar tijdsloten.

    tijdsloten: lijst van uren waarop ambulanten welkom zijn (bv. [12,13,16,17])
    fractie:    deel van ambulanten dat naar tijdsloten gestuurd wordt (0.0–1.0)

    Logica:
    - Piekuren (niet in tijdsloten): aankomstrate × (1 - fractie)
    - Tijdsloten: baseline + evenredig deel van herverdeelde patiënten
    """
    if fractie == 0 or not tijdsloten:
        return dict(AANKOMST_AMBULANT)

    nieuw = dict(AANKOMST_AMBULANT)
    piekuren = [h for h in AANKOMST_AMBULANT if h not in tijdsloten]

    totaal_herverdeeld = sum(AANKOMST_AMBULANT[h] for h in piekuren) * fractie
    extra_per_slot = totaal_herverdeeld / len(tijdsloten)

    for h in piekuren:
        nieuw[h] = AANKOMST_AMBULANT[h] * (1 - fractie)
    for h in tijdsloten:
        nieuw[h] = AANKOMST_AMBULANT.get(h, 0.0) + extra_per_slot

    return nieuw


# ══════════════════════════════════════════════════════════════════════════════
# 3. PATIENT-DATAMODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Patient:
    """
    Bijgehouden info per patiënt doorheen de simulatie.
    Tijden zijn in minuten vanaf het begin van de simulatie.
    """
    id:            int
    aankomst:      float          # tijdstip aankomst
    type:          str            # 'A', 'H', 'D'
    categorie:     str            # onderzoekscategorie
    wacht_start:   float = 0.0   # tijdstip begin wachten (= aankomst als wachtkamer in)
    exam_start:    float = 0.0   # tijdstip begin onderzoek
    exam_einde:    float = 0.0   # tijdstip einde onderzoek
    zaal:          str  = ''     # toegewezen zaal
    gebalk:        bool = False  # True = patiënt verliet wachtkamer

    @property
    def wachttijd(self) -> float:
        return max(0.0, self.exam_start - self.aankomst)

    @property
    def exam_duur(self) -> float:
        return max(0.0, self.exam_einde - self.exam_start)

    @property
    def totale_tijd(self) -> float:
        return max(0.0, self.exam_einde - self.aankomst)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATIEKLASSE
# ══════════════════════════════════════════════════════════════════════════════

class RXSimulatie:
    """
    Discrete-event simulatie van de RX-afdeling.

    Gebruik:
        config = { 'actieve_zalen': ['G5','G15','G3','G4'], ... }
        sim = RXSimulatie(config, seed=42)
        resultaten = sim.run()
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed   = seed

    def run(self) -> dict:
        np.random.seed(self.seed)

        # ── Configuratie uitlezen ─────────────────────────────────────────────
        actieve_zalen   = self.config.get('actieve_zalen',    DEFAULT_ZALEN)
        n_technici      = self.config.get('n_technici',        DEFAULT_TECHNICI)
        wk_cap          = self.config.get('wachtkamer_cap',    DEFAULT_WACHTKAMER)
        balk_drempel    = self.config.get('balk_drempel_min',  60)
        sim_uren        = self.config.get('sim_uren',          12)
        schaal_A        = self.config.get('schaal_ambulant',   1.0)
        schaal_H        = self.config.get('schaal_hosp',       1.0)
        tijdsloten      = self.config.get('tijdsloten',        [])
        slot_fractie    = self.config.get('slot_fractie',      0.0)
        fifo            = self.config.get('fifo',              True)

        # Ambulant profiel aanpassen op basis van tijdsloten
        ambulant_profiel = bouw_ambulant_profiel(tijdsloten, slot_fractie)

        # ── SimPy resources ───────────────────────────────────────────────────
        env = simpy.Environment()

        # Één resource per actieve zaal (capacity=1, want 1 patiënt per zaal)
        zaal_resources = {
            zaal: simpy.Resource(env, capacity=1)
            for zaal in actieve_zalen
        }

        # Technici: gedeelde pool (aanname A2: 1 technicus per onderzoek)
        # Prioriteit of FIFO afhankelijk van configuratie
        if fifo:
            technici = simpy.Resource(env, capacity=n_technici)
        else:
            # H krijgt lichte voorrang (zie aanname A1)
            technici = simpy.PriorityResource(env, capacity=n_technici)

        wachtkamer = simpy.Container(env, capacity=wk_cap, init=0)

        patienten:  List[Patient] = []
        wachtrij_log: List[dict] = []

        # ── Patiëntproces ─────────────────────────────────────────────────────
        def patient_proces(patient: Patient):
            """
            Levenscyclus van één patiënt:
            1. Aankomst → probeer wachtkamer te betreden
            2. Wachten op geschikte vrije zaal + vrije technicus
            3. Onderzoek uitvoeren
            4. Vertrek
            """
            # Wachtkamer vol → patiënt kan niet wachten (balking op de deur)
            if wachtkamer.level >= wachtkamer.capacity:
                patient.gebalk = True
                patienten.append(patient)
                return

            yield wachtkamer.put(1)
            patient.wacht_start = env.now
            gebalk_vlag = [False]

            # Balk-timer: patiënt verlaat wachtkamer na balk_drempel minuten
            def balk_timer():
                try:
                    yield env.timeout(balk_drempel)
                    if patient.exam_start == 0.0:
                        gebalk_vlag[0] = True
                except simpy.Interrupt:
                    pass  # onderzoek begon voor timer afliep

            balk_proc = env.process(balk_timer())

            # Zoek geschikte zalen voor dit onderzoekstype
            geschikte_zalen = [
                z for z in ZAAL_VOORKEUR.get(patient.categorie, actieve_zalen)
                if z in actieve_zalen
            ]
            if not geschikte_zalen:
                geschikte_zalen = actieve_zalen  # fallback: alle actieve zalen

            # Wacht op de eerste beschikbare geschikte zaal
            # Aanpak: vraag tegelijk aan alle geschikte zalen, neem de eerste
            def probeer_zalen():
                """Vraagt resources aan in volgorde van voorkeur."""
                aanvragen = []
                for zaal_naam in geschikte_zalen:
                    req = zaal_resources[zaal_naam].request()
                    aanvragen.append((zaal_naam, req))

                # Wacht tot één zaal beschikbaar is
                events = [req for _, req in aanvragen]
                resultaat = yield simpy.AnyOf(env, events)

                # Vrijgeven van zalen die we niet gebruiken
                gekozen_zaal = None
                gekozen_req  = None
                for zaal_naam, req in aanvragen:
                    if req in resultaat:
                        if gekozen_zaal is None:
                            gekozen_zaal = zaal_naam
                            gekozen_req  = req
                        else:
                            zaal_resources[zaal_naam].release(req)
                    else:
                        try:
                            zaal_resources[zaal_naam].release(req)
                        except Exception:
                            pass

                return gekozen_zaal, gekozen_req

            # Technicus aanvragen (FIFO of prioriteit)
            if fifo:
                tech_req = technici.request()
            else:
                prioriteit = {'H': 1, 'D': 2, 'A': 3}[patient.type]
                tech_req = technici.request(priority=prioriteit)

            # Wacht op technicus én geschikte zaal
            zaal_proc = env.process(probeer_zalen())
            yield tech_req & zaal_proc

            gekozen_zaal, zaal_req = zaal_proc.value

            if gebalk_vlag[0]:
                # Patiënt is al weg — resources vrijgeven
                patient.gebalk = True
                technici.release(tech_req)
                if zaal_req and gekozen_zaal:
                    zaal_resources[gekozen_zaal].release(zaal_req)
                yield wachtkamer.get(1)
                patienten.append(patient)
                return

            if balk_proc.is_alive:
                balk_proc.interrupt()

            # Onderzoek uitvoeren
            patient.exam_start = env.now
            patient.zaal       = gekozen_zaal or ''
            duratie            = sample_duratie(patient.categorie)
            yield env.timeout(duratie)
            patient.exam_einde = env.now

            # Resources vrijgeven
            technici.release(tech_req)
            if zaal_req and gekozen_zaal:
                zaal_resources[gekozen_zaal].release(zaal_req)
            yield wachtkamer.get(1)

            patienten.append(patient)

        # ── Wachtrij monitor ──────────────────────────────────────────────────
        def monitor():
            """Logt de wachtkamerbezetting elke 5 minuten."""
            while True:
                bezig = sum(
                    1 for z in actieve_zalen
                    if zaal_resources[z].count > 0
                )
                wachtrij_log.append({
                    'tijd':          env.now,
                    'wachtkamer':    wachtkamer.level,
                    'zalen_bezig':   bezig,
                    'tech_bezig':    technici.count,
                })
                yield env.timeout(5)

        # ── Aankomstgeneratoren ───────────────────────────────────────────────
        def aankomst_generator(profiel: dict, type_code: str, schaal: float):
            """Genereert patiënten via een Poisson-proces per uur."""
            pid_offset = {'A': 0, 'H': 1_000_000, 'D': 2_000_000}[type_code]
            pid = pid_offset
            while True:
                rate = get_aankomstrate(profiel, env.now, schaal)
                iat  = np.random.exponential(60 / rate) if rate > 0 else 60
                yield env.timeout(iat)
                pid += 1
                p = Patient(
                    id=pid,
                    aankomst=env.now,
                    type=type_code,
                    categorie=sample_categorie(),
                )
                env.process(patient_proces(p))

        # Start alle processen
        env.process(aankomst_generator(ambulant_profiel, 'A', schaal_A))
        env.process(aankomst_generator(AANKOMST_HOSP,    'H', schaal_H))
        env.process(aankomst_generator(AANKOMST_DAG,     'D', 1.0))
        env.process(monitor())
        env.run(until=sim_uren * 60)

        return self._bereken_metrics(patienten, wachtrij_log, ambulant_profiel)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _bereken_metrics(self, patienten, wachtrij_log, ambulant_profiel) -> dict:
        afgehandeld = [p for p in patienten if not p.gebalk and p.exam_einde > 0]
        gebalkt     = [p for p in patienten if p.gebalk]

        if not afgehandeld:
            return {'fout': 'Geen patiënten afgehandeld', 'config': self.config}

        type_labels = {'A': 'Ambulant', 'H': 'Gehospitaliseerd', 'D': 'Dagziekenhuis'}

        df = pd.DataFrame([{
            'wachttijd':    p.wachttijd,
            'exam_duur':    p.exam_duur,
            'totale_tijd':  p.totale_tijd,
            'type':         p.type,
            'type_label':   type_labels[p.type],
            'categorie':    p.categorie,
            'zaal':         p.zaal,
            'uur':          int(p.aankomst / 60) % 24,
        } for p in afgehandeld])

        qdf = pd.DataFrame(wachtrij_log)
        cap = self.config.get('wachtkamer_cap', DEFAULT_WACHTKAMER)

        # Coëfficiënt van variatie werkdruk (maatstaf voor smoothing)
        vol_per_uur  = df.groupby('uur').size()
        werkdruk_cv  = (vol_per_uur.std() / vol_per_uur.mean() * 100
                        if len(vol_per_uur) > 1 else 0)

        return {
            'totaal':           len(patienten),
            'afgehandeld':      len(afgehandeld),
            'gebalkt':          len(gebalkt),
            'balk_rate':        len(gebalkt) / max(len(patienten), 1) * 100,
            'gem_wacht':        df['wachttijd'].mean(),
            'mediaan_wacht':    df['wachttijd'].median(),
            'p90_wacht':        df['wachttijd'].quantile(0.90),
            'p95_wacht':        df['wachttijd'].quantile(0.95),
            'gem_totaal':       df['totale_tijd'].mean(),
            'werkdruk_cv':      werkdruk_cv,
            'per_type':         df.groupby('type_label')['wachttijd']
                                  .agg(['mean','median','count']).to_dict(),
            'per_categorie':    df.groupby('categorie')['wachttijd'].mean().to_dict(),
            'per_uur':          df.groupby('uur')['wachttijd'].mean().to_dict(),
            'volume_per_uur':   df.groupby('uur').size().to_dict(),
            'volume_type_uur':  df.groupby(['uur','type_label']).size()
                                  .unstack(fill_value=0).to_dict(),
            'per_zaal':         df.groupby('zaal')['wachttijd'].mean().to_dict() if 'zaal' in df.columns else {},
            'zaal_gebruik':     df.groupby('zaal').size().to_dict() if 'zaal' in df.columns else {},
            'wachtrij_df':      qdf,
            'patient_df':       df,
            'ambulant_profiel': ambulant_profiel,
            'config':           self.config,
        }
