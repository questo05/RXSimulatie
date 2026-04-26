"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RX WACHTLIJSTSIMULATIE — UZ Gasthuisberg                                   ║
║  Discrete-event simulatie via SimPy                                          ║
║  Geparametriseerd met KWS-data 2021-2025 (165.259 contacten, 337k acta)     ║
╚══════════════════════════════════════════════════════════════════════════════╝

ASSUMPTIES & KEUZES
───────────────────
A1. FIFO-wachtrij: geen prioriteit tussen patiënten (first-in-first-out).
    Optioneel uitschakelbaar: gehospitaliseerden (H) krijgen dan lichte
    voorrang omdat een verpleegkundige op hen wacht.

A2. Resources: elke patiënt heeft 1 zaal + 1 technicus nodig tegelijk.

A3. Geschiktheidsmatrix: G5=thorax, G15=full-length, G2/G3/G4=gemengd,
    G1=backup. Gebaseerd op empirisch gebruik 2021-2025. Patiënt wacht
    op geschikte zaal, gaat niet naar ongeschikte.

A4. Exam duraties: geschat via tijdsverschil opeenvolgende acta per zaal.
    Echte begin/eindtijden ontbreken in de data.

A5. Balking: patiënten die langer dan X minuten wachten verlaten de
    wachtkamer. Wachtkamer vol = directe afwijzing.

A6. Aankomstproces: Poisson-proces per uur op basis van gemiddeld profiel
    werkdagen 2021-2025.

A7. Tijdsloten — enkel STUURBARE ambulanten:
    Slechts 14.6% van ambulante patiënten heeft geen andere consultatie
    op dezelfde dag (kolom 'contact detail receptie uur' is leeg).
    Alleen deze groep (gem. 14.8/dag) is stuurbaar naar tijdsloten.
    De overige 85.4% (met consultatie, gem. 86.7/dag) komen op het
    normale patroon toe en zijn niet stuurbaar.

A8. Drukte-niveaus gebaseerd op dagvolume percentiel (KWS 2021-2025):
    Rustig  = P25 (~111 patiënten/dag), schaalfactor ×0.85
    Normaal = P50 (~128 patiënten/dag), schaalfactor ×1.00
    Druk    = P75 (~150 patiënten/dag), schaalfactor ×1.17
    Erg druk= P90 (~172 patiënten/dag), schaalfactor ×1.34
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Aankomstprofielen ─────────────────────────────────────────────────────────
# Gemiddeld patiënten/uur per werkdag, bron: KWS-data 2021-2025 (1266 werkdagen)

# Ambulant STUURBAAR (AD Type A, lege consultatie) — gem. 14.8/dag
AANKOMST_AMBULANT_STUURBAAR = {
    6: 0.07,  7: 0.25,  8: 1.14,  9: 1.67, 10: 2.03,
    11: 1.84, 12: 1.28, 13: 1.47, 14: 1.86, 15: 1.72,
    16: 1.24, 17: 0.22,
}

# Ambulant NIET-STUURBAAR (AD Type A, heeft consultatie) — gem. 86.7/dag
AANKOMST_AMBULANT_VAST = {
    6: 0.06,  7: 3.22,  8: 10.25,  9: 12.55, 10: 12.01,
    11: 7.65, 12: 5.75, 13:  9.37, 14: 11.46, 15:  8.81,
    16: 4.58, 17: 0.99,
}

# Gehospitaliseerd (AD Type H) — gem. 23.1/dag
AANKOMST_HOSP = {
    5: 0.07, 6: 11.83, 7: 0.52, 8: 1.49, 9: 1.33, 10: 1.57,
    11: 1.21, 12: 0.65, 13: 1.63, 14: 1.30, 15: 0.81,
    16: 0.52, 17: 0.13,
}

# Dagziekenhuis (AD Type D) — gem. 6.0/dag
AANKOMST_DAG = {
    7: 0.08, 8: 0.92, 9: 1.06, 10: 0.92, 11: 0.53,
    12: 0.44, 13: 0.76, 14: 0.59, 15: 0.35, 16: 0.18, 17: 0.04,
}

# ── Drukte-niveaus ────────────────────────────────────────────────────────────
# Gebaseerd op dagvolume-percentielen KWS 2021-2025
# Schaalfactor past het aankomstvolume aan voor alle types
DRUKTE_NIVEAUS = {
    'Rustig (P25, ~111/dag)':    0.85,
    'Normaal (P50, ~128/dag)':   1.00,
    'Druk (P75, ~150/dag)':      1.17,
    'Erg druk (P90, ~172/dag)':  1.34,
}

# ── Onderzoekscategorieën ─────────────────────────────────────────────────────
CATEGORIE_MIX = {
    'cat_thorax':        0.2727,
    'cat_extremiteiten': 0.2919,
    'cat_tafelwerk':     0.1555,
    'cat_long_length':   0.0680,
    'cat_overig':        0.1119,
}

def categorie_van_omschrijving(omschrijving: str) -> str:
    if pd.isna(omschrijving): return 'cat_overig'
    s = str(omschrijving).upper()
    if 'THORAX' in s or 'LONG' in s: return 'cat_thorax'
    if any(x in s for x in ['FULL','SPINE','LENGTE','EOS']): return 'cat_long_length'
    if any(x in s for x in ['WERVEL','NEK','RUG','BEKKEN','SACRUM','HEUP','HIP','ABDOMEN','BUIK']):
        return 'cat_tafelwerk'
    if any(x in s for x in ['HAND','POLS','ARM','ELBOW','VINGER','SCHOUDER',
                              'KNIE','VOET','ENKEL','BEEN','FEMUR']):
        return 'cat_extremiteiten'
    return 'cat_overig'

# ── Exam duraties (minuten, lognormaal gefit op mediaan+std) ──────────────────
EXAM_DURATIE = {
    'cat_thorax':        {'median': 7,  'std': 8.8},
    'cat_extremiteiten': {'median': 8,  'std': 9.3},
    'cat_tafelwerk':     {'median': 7,  'std': 9.4},
    'cat_long_length':   {'median': 11, 'std': 13.0},
    'cat_overig':        {'median': 7,  'std': 9.5},
}

# ── Geschiktheidsmatrix zalen/categorieën ─────────────────────────────────────
GESCHIKTHEID = {
    'G5':  {'cat_thorax': True,  'cat_extremiteiten': False, 'cat_tafelwerk': False, 'cat_long_length': False, 'cat_overig': False},
    'G15': {'cat_thorax': False, 'cat_extremiteiten': False, 'cat_tafelwerk': False, 'cat_long_length': True,  'cat_overig': False},
    'G3':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': False, 'cat_overig': True},
    'G4':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': False, 'cat_overig': True},
    'G2':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': False, 'cat_overig': True},
    'G1':  {'cat_thorax': True,  'cat_extremiteiten': True,  'cat_tafelwerk': True,  'cat_long_length': True,  'cat_overig': True},
}

ZAAL_VOORKEUR = {
    'cat_thorax':        ['G5', 'G3', 'G4', 'G2', 'G1'],
    'cat_long_length':   ['G15', 'G1'],
    'cat_extremiteiten': ['G3', 'G4', 'G2', 'G1'],
    'cat_tafelwerk':     ['G3', 'G4', 'G2', 'G1'],
    'cat_overig':        ['G3', 'G4', 'G2', 'G1'],
}

DEFAULT_ZALEN      = ['G5', 'G15', 'G3', 'G4', 'G2']
DEFAULT_TECHNICI   = 6
DEFAULT_WACHTKAMER = 30

# Simulatievenster: 7u tot 18u = 11 uur
SIM_START_UUR = 7
SIM_EINDE_UUR = 18
SIM_DUUR_MIN  = (SIM_EINDE_UUR - SIM_START_UUR) * 60  # 660 minuten


# ══════════════════════════════════════════════════════════════════════════════
# 2. HULPFUNCTIES
# ══════════════════════════════════════════════════════════════════════════════

def sample_categorie() -> str:
    cats  = list(CATEGORIE_MIX.keys())
    probs = np.array(list(CATEGORIE_MIX.values()))
    return np.random.choice(cats, p=probs / probs.sum())


def sample_duratie(categorie: str) -> float:
    """Correcte lognormale: mu = ln(mediaan) zodat mediaan = exp(mu)."""
    p = EXAM_DURATIE[categorie]
    median, std = p['median'], p['std']
    sigma = np.sqrt(np.log(1 + (std / median) ** 2))
    mu    = np.log(median)
    return max(2.0, np.random.lognormal(mu, sigma))


def get_rate(profiel: dict, sim_minuten: float, schaal: float = 1.0) -> float:
    """Zet simulatietijd (minuten vanaf SIM_START_UUR) om naar uur van de dag."""
    uur = int(SIM_START_UUR + sim_minuten / 60) % 24
    return profiel.get(uur, 0.0) * schaal


def bouw_stuurbaar_profiel_en_slots(tijdsloten_halfuur: List[float], fractie: float, schaal: float):
    """
    Haalt een fractie van de piek-patiënten uit de Poisson-rates en 
    wijst ze deterministisch toe aan specifieke tijdsloten.
    """
    if fractie == 0 or not tijdsloten_halfuur:
        return dict(AANKOMST_AMBULANT_STUURBAAR), []

    slot_uren = sorted(set(int(h) for h in tijdsloten_halfuur))
    alle_uren = list(AANKOMST_AMBULANT_STUURBAAR.keys())
    piek_uren = [u for u in alle_uren if u not in slot_uren]

    # Bereken hoeveel patiënten (verwachtingswaarde) we sturen naar de daluren
    verwacht_te_verplaatsen = sum(AANKOMST_AMBULANT_STUURBAAR[u] for u in piek_uren) * fractie * schaal
    
    # We zetten de verwachtingswaarde om in een discreet, werkelijk aantal patiënten voor deze run
    n_gepland = np.random.poisson(verwacht_te_verplaatsen)

    profiel_buiten = {}
    for u in alle_uren:
        if u in slot_uren:
            # Uren waar slots zijn behouden hun natuurlijke walk-in flow
            profiel_buiten[u] = AANKOMST_AMBULANT_STUURBAAR[u]
        else:
            # Piekuren verliezen het afgeroomde volume
            profiel_buiten[u] = AANKOMST_AMBULANT_STUURBAAR[u] * (1 - fractie)

    # Wijs dit aantal patiënten gelijkmatig (of random) toe aan de geselecteerde tijdsloten
    geplande_tijden = []
    if n_gepland > 0 and tijdsloten_halfuur:
        geplande_tijden = sorted(list(np.random.choice(tijdsloten_halfuur, size=n_gepland)))

    return profiel_buiten, geplande_tijden


# ══════════════════════════════════════════════════════════════════════════════
# 3. PATIENT-DATAMODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Patient:
    id:           int
    aankomst:     float    # minuten vanaf start simulatie (= 7u)
    type:         str      # 'A_vast', 'A_stuurbaar', 'H', 'D'
    categorie:    str
    wacht_start:  float = 0.0
    exam_start:   float = 0.0
    exam_einde:   float = 0.0
    zaal:         str   = ''
    gebalk:       bool  = False

    @property
    def aankomst_uur(self) -> float:
        """Werkelijk uur van de dag (bv. 8.5 = 8u30)."""
        return SIM_START_UUR + self.aankomst / 60

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
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed   = seed

    def run(self) -> dict:
        """
        COMMON RANDOM NUMBERS (CRN) — belangrijk voor eerlijke scenario-vergelijking
        ─────────────────────────────────────────────────────────────────────────────
        Het doel van de simulatie is de TOEGEVOEGDE WAARDE van tijdsloten aantonen.
        Dat vereist dat alle niet-stuurbare patiënten (A_vast, H, D) en de exam-
        duraties identiek zijn in elk scenario — alleen de stuurbare ambulanten
        veranderen. We realiseren dit via geïsoleerde RNG's per stroom:

          rng_vast / rng_hosp / rng_dag  → ALTIJD zelfde seed → zelfde patiënten
          rng_duratie                    → ALTIJD zelfde seed → zelfde exam-duraties
          rng_stuurbaar                  → past aan op basis van scenario (normaal
                                           Poisson buiten tijdsloten)
          rng_slots                      → enkel voor de geplande tijdslot-patiënten

        Zo is elk verschil in resultaat UITSLUITEND toe te schrijven aan de
        tijdsloten, niet aan toeval in andere stromen.
        """
        actieve_zalen = self.config.get('actieve_zalen',     DEFAULT_ZALEN)
        n_technici    = self.config.get('n_technici',         DEFAULT_TECHNICI)
        wk_cap        = self.config.get('wachtkamer_cap',     DEFAULT_WACHTKAMER)
        balk_drempel  = self.config.get('balk_drempel_min',   180)
        schaal        = self.config.get('schaal',             1.0)
        tijdsloten    = self.config.get('tijdsloten',         [])
        slot_fractie  = self.config.get('slot_fractie',       0.0)
        fifo          = self.config.get('fifo',               True)

        # ── CRN: geïsoleerde RNG's ────────────────────────────────────────────
        # Niet-stuurbare stromen: vaste seed → identiek in elk scenario
        rng_vast      = np.random.RandomState(self.seed)
        rng_hosp      = np.random.RandomState(self.seed + 1)
        rng_dag       = np.random.RandomState(self.seed + 2)
        rng_duratie   = np.random.RandomState(self.seed + 3)  # exam-duraties
        # Stuurbare stromen: mogen variëren per scenario
        rng_stuurbaar = np.random.RandomState(self.seed + 4)
        rng_slots     = np.random.RandomState(self.seed + 5)

        # ── Stuurbaar profiel opbouwen ────────────────────────────────────────
        # Logica:
        #   - fractie van het totale stuurbare dagvolume wordt weggehaald
        #     uit het Poisson-proces en deterministisch gepland in tijdsloten
        #   - de rest blijft als Poisson-proces (proportioneel verminderd per uur)
        #   - profiel_buiten: rates voor het Poisson-proces (patiënten/uur)
        #   - profiel_slot_rates: rates voor de grafiek (ook patiënten/uur,
        #     berekend als n_gepland / duur_slot_uur)

        profiel_buiten   = dict(AANKOMST_AMBULANT_STUURBAAR)
        geplande_tijden  = []
        profiel_slot_rates = {}  # uur → rate (patiënten/uur) voor grafiek

        if slot_fractie > 0 and tijdsloten:
            alle_uren = list(AANKOMST_AMBULANT_STUURBAAR.keys())

            # Totaal verwacht te verplaatsen per dag (als gemiddelde, niet discreet)
            totaal_per_dag        = sum(AANKOMST_AMBULANT_STUURBAAR.values())
            totaal_herverdeeld    = totaal_per_dag * slot_fractie * schaal

            # Trek een discreet aantal patiënten voor deze simulatierun
            n_gepland = rng_slots.poisson(totaal_herverdeeld)

            # Elk uur verliest proportioneel dezelfde fractie uit het Poisson-proces
            profiel_buiten = {
                u: AANKOMST_AMBULANT_STUURBAAR[u] * (1 - slot_fractie)
                for u in alle_uren
            }

            # Wijs n_gepland patiënten toe aan tijdsloten (uniform over actieve slots)
            if n_gepland > 0:
                gekozen_slots   = sorted(rng_slots.choice(tijdsloten, size=n_gepland))
                geplande_tijden = gekozen_slots

                # Bereken rate per uur voor de grafiek (aantallen → patiënten/uur)
                # Een halfuurslot heeft een "breedte" van 0.5 uur
                slot_halven = {}
                for h in tijdsloten:
                    uur = int(h)
                    slot_halven[uur] = slot_halven.get(uur, 0) + 1

                for uur, n_halven in slot_halven.items():
                    # Gemiddeld verwacht aantal in dit uur = n_gepland × (n_halven / totaal_slots)
                    aandeel = n_halven / len(tijdsloten)
                    profiel_slot_rates[uur] = (totaal_herverdeeld * aandeel)

        env = simpy.Environment()
        zaal_res   = {z: simpy.Resource(env, capacity=1) for z in actieve_zalen}
        technici   = (simpy.Resource(env, capacity=n_technici) if fifo
                      else simpy.PriorityResource(env, capacity=n_technici))
        wachtkamer = simpy.Container(env, capacity=wk_cap, init=0)

        patienten:    List[Patient] = []
        wachtrij_log: List[dict]   = []

        def lokaal_sample_categorie(rng) -> str:
            cats  = list(CATEGORIE_MIX.keys())
            probs = np.array(list(CATEGORIE_MIX.values()))
            return rng.choice(cats, p=probs / probs.sum())

        def lokaal_sample_duratie(categorie: str) -> float:
            """
            Gebruikt altijd rng_duratie -> zelfde exam-duraties in elk scenario (CRN).
            Correcte lognormale parametrisatie: mediaan = exp(mu) -> mu = ln(mediaan).
            Gemiddelden liggen hoger dan mediaan door lange staart van de lognormale.
            """
            p = EXAM_DURATIE[categorie]
            median, std = p['median'], p['std']
            sigma = np.sqrt(np.log(1 + (std / median) ** 2))
            mu    = np.log(median)
            return max(2.0, rng_duratie.lognormal(mu, sigma))

        def patient_proces(patient: Patient):
            if wachtkamer.level >= wachtkamer.capacity:
                patient.gebalk = True
                patienten.append(patient)
                return

            yield wachtkamer.put(1)
            patient.wacht_start = env.now
            gebalk_vlag = [False]

            def balk_timer():
                try:
                    yield env.timeout(balk_drempel)
                    if patient.exam_start == 0.0:
                        gebalk_vlag[0] = True
                except simpy.Interrupt:
                    pass

            balk_proc = env.process(balk_timer())

            geschikte = (
                [z for z in ZAAL_VOORKEUR.get(patient.categorie, actieve_zalen)
                 if z in actieve_zalen] or actieve_zalen
            )

            def probeer_zalen():
                aanvragen = [(z, zaal_res[z].request()) for z in geschikte]
                resultaat = yield simpy.AnyOf(env, [r for _, r in aanvragen])
                
                gekozen_zaal = gekozen_req = None
                
                for z, req in aanvragen:
                    if req in resultaat:
                        # Deze request is direct gehonoreerd
                        if gekozen_zaal is None:
                            # Dit is de eerste toegewezen zaal, we houden deze!
                            gekozen_zaal, gekozen_req = z, req
                        else:
                            # Er was nóg een zaal op exact hetzelfde moment vrij.
                            # Omdat we al een zaal gekozen hebben, geven we deze direct weer vrij.
                            zaal_res[z].release(req)
                    else:
                        # Deze request staat nog te wachten in de wachtrij van de zaal.
                        # We annuleren het Request-object zodat de zaal niet later blokkeert.
                        req.cancel()
                        
                return gekozen_zaal, gekozen_req
            
            tech_req = (technici.request() if fifo else
                        technici.request(
                            priority={'H': 1, 'D': 2,
                                      'A_vast': 3, 'A_stuurbaar': 3}[patient.type]))
            zaal_proc = env.process(probeer_zalen())
            yield tech_req & zaal_proc
            gekozen_zaal, zaal_req = zaal_proc.value

            if gebalk_vlag[0]:
                patient.gebalk = True
                technici.release(tech_req)
                if zaal_req:
                    zaal_res[gekozen_zaal].release(zaal_req)
                yield wachtkamer.get(1)
                patienten.append(patient)
                return

            if balk_proc.is_alive:
                balk_proc.interrupt()

            patient.exam_start = env.now
            patient.zaal       = gekozen_zaal or ''
            yield env.timeout(lokaal_sample_duratie(patient.categorie))
            patient.exam_einde = env.now
            technici.release(tech_req)
            if zaal_req:
                zaal_res[gekozen_zaal].release(zaal_req)
            yield wachtkamer.get(1)
            patienten.append(patient)

        def monitor():
            while True:
                wachtrij_log.append({
                    'tijd':        env.now,
                    'wachtkamer':  wachtkamer.level,
                    'zalen_bezig': sum(1 for z in actieve_zalen
                                       if zaal_res[z].count > 0),
                    'tech_bezig':  technici.count,
                })
                yield env.timeout(5)

        def aankomst_gen(profiel: dict, type_code: str, schaal_factor: float, rng):
            pid_offset = {
                'A_vast': 0, 'A_stuurbaar': 500_000,
                'H': 1_000_000, 'D': 2_000_000,
            }[type_code]
            pid = pid_offset
            
            while True:
                # 1. Haal de rate op voor de huidige minuut
                rate_per_uur = get_rate(profiel, env.now, schaal_factor)
                rate_per_minuut = rate_per_uur / 60.0
                
                # 2. Check of er in deze specifieke minuut een patiënt toekomt
                # (Bernoulli-benadering van een Poisson-proces, perfect voor deze volumes)
                if rate_per_minuut > 0 and rng.random() < rate_per_minuut:
                    pid += 1
                    env.process(patient_proces(Patient(
                        id=pid, aankomst=env.now, type=type_code,
                        categorie=lokaal_sample_categorie(rng),
                    )))
                    
                # 3. Schuif de simulatieklok betrouwbaar met exact 1 minuut op
                yield env.timeout(1.0)

        def aankomst_gen_slots(tijden_uren: List[float], type_code: str, rng):
            """
            Deterministisch aankomstgenerator voor geplande tijdslot-patiënten.
            Patiënten komen toe op hun geplande tijdstip ± 5 minuten (normaal
            verdeeld) — realistisch want mensen zijn nooit exact op tijd.
            """
            pid = 600_000
            tijden_sim_min = [(t - SIM_START_UUR) * 60 for t in tijden_uren]
            # Gaussian jitter ±5 min → realistisch aankomstgedrag
            werkelijke_tijden = sorted(
                max(0.0, t + rng.normal(0, 5)) for t in tijden_sim_min
            )
            for w_tijd in werkelijke_tijden:
                wachten = w_tijd - env.now
                if wachten > 0:
                    yield env.timeout(wachten)
                pid += 1
                env.process(patient_proces(Patient(
                    id=pid, aankomst=env.now, type=type_code,
                    categorie=lokaal_sample_categorie(rng),
                )))

        # ── Start generatoren ─────────────────────────────────────────────────
        # Niet-stuurbaar: vaste RNG → identiek in elk scenario
        env.process(aankomst_gen(AANKOMST_AMBULANT_VAST, 'A_vast',      schaal, rng_vast))
        env.process(aankomst_gen(AANKOMST_HOSP,          'H',           schaal, rng_hosp))
        env.process(aankomst_gen(AANKOMST_DAG,           'D',           schaal, rng_dag))
        # Stuurbaar buiten tijdsloten: Poisson op verminderd profiel
        env.process(aankomst_gen(profiel_buiten,         'A_stuurbaar', schaal, rng_stuurbaar))
        # Stuurbaar in tijdsloten: deterministisch
        if geplande_tijden:
            env.process(aankomst_gen_slots(geplande_tijden, 'A_stuurbaar', rng_slots))

        env.process(monitor())
        env.run(until=SIM_DUUR_MIN)

        return self._bereken_metrics(
            patienten, wachtrij_log, profiel_buiten, profiel_slot_rates
        )
    
    def _bereken_metrics(self, patienten, wachtrij_log,
                         profiel_vast, profiel_slot) -> dict:
        afgehandeld = [p for p in patienten if not p.gebalk and p.exam_einde > 0]
        gebalkt     = [p for p in patienten if p.gebalk]

        if not afgehandeld:
            return {'fout': 'Geen patiënten afgehandeld', 'config': self.config}

        # Veranderd naar "zonder consultatie"
        type_labels = {
            'A_vast':      'Ambulant (met consultatie)',
            'A_stuurbaar': 'Ambulant (zonder consultatie)',
            'H':           'Gehospitaliseerd',
            'D':           'Dagziekenhuis',
        }
        type_kleuren = {
            'Ambulant (met consultatie)':    '#60a5fa',
            'Ambulant (zonder consultatie)': '#93c5fd',
            'Gehospitaliseerd':              '#f87171',
            'Dagziekenhuis':                 '#34d399',
        }

        df = pd.DataFrame([{
            'wachttijd':   p.wachttijd,
            'exam_duur':   p.exam_duur,
            'totale_tijd': p.totale_tijd,
            'type':        p.type,
            'label':       type_labels[p.type],
            'categorie':   p.categorie,
            'zaal':        p.zaal,
            'uur_float':   p.aankomst_uur,
            'uur':         int(p.aankomst_uur),
        } for p in afgehandeld])

        # Aankomsten = ALLE patienten (ook gebalkt), want dit toont de werkelijke
        # instroom. volume_per_uur (afgehandeld) en aankomst_per_uur (totaal) zijn
        # twee verschillende metrics die elk iets anders vertellen.
        df_alle = pd.DataFrame([{
            'type':    p.type,
            'label':   type_labels[p.type],
            'uur':     int(p.aankomst_uur),
            'gebalk':  p.gebalk,
        } for p in patienten])

        aankomst_per_uur       = df_alle.groupby('uur').size().to_dict()
        aankomst_label_uur     = (df_alle.groupby(['uur','label']).size()
                                  .unstack(fill_value=0).to_dict())

        qdf = pd.DataFrame(wachtrij_log)
        qdf['klok_uur'] = SIM_START_UUR + qdf['tijd'] / 60

        # Werkdruk CV op basis van AANKOMSTEN (niet afgehandeld) — eerlijker maatstaf
        vol_per_uur = pd.Series(aankomst_per_uur)
        werkdruk_cv = (vol_per_uur.std() / vol_per_uur.mean() * 100
                       if len(vol_per_uur) > 1 else 0)

        alle_uren = sorted(set(list(profiel_vast.keys()) + list(profiel_slot.keys())))
        ambulant_profiel_totaal = {
            u: profiel_vast.get(u, 0) + profiel_slot.get(u, 0)
            for u in alle_uren
        }

        return {
            'totaal':            len(patienten),
            'afgehandeld':       len(afgehandeld),
            'gebalkt':           len(gebalkt),
            'balk_rate':         len(gebalkt) / max(len(patienten), 1) * 100,
            'gem_wacht':         df['wachttijd'].mean(),
            'mediaan_wacht':     df['wachttijd'].median(),
            'p90_wacht':         df['wachttijd'].quantile(0.90),
            'p95_wacht':         df['wachttijd'].quantile(0.95),
            'gem_totaal':        df['totale_tijd'].mean(),
            'werkdruk_cv':       werkdruk_cv,
            'per_label':         df.groupby('label')['wachttijd']
                                   .agg(['mean','median','count']).to_dict(),
            'per_categorie':     df.groupby('categorie')['wachttijd'].mean().to_dict(),
            'per_uur':           df.groupby('uur')['wachttijd'].mean().to_dict(),
            # Afgehandeld per uur (wat de zalen verwerkt hebben)
            'volume_per_uur':    df.groupby('uur').size().to_dict(),
            'volume_label_uur':  df.groupby(['uur','label']).size()
                                   .unstack(fill_value=0).to_dict(),
            # Aankomsten per uur (de werkelijke instroom, incl. gebalkt)
            # Dit is de correcte metric om het effect van tijdsloten te tonen:
            # buiten tijdsloten MINDER aankomsten, in tijdsloten MEER aankomsten
            'aankomst_per_uur':      aankomst_per_uur,
            'aankomst_label_uur':    aankomst_label_uur,
            'per_zaal':          df.groupby('zaal')['wachttijd'].mean().to_dict(),
            'zaal_gebruik':      df.groupby('zaal').size().to_dict(),
            'wachtrij_df':       qdf,
            'patient_df':        df,
            'type_kleuren':      type_kleuren,
            'profiel_vast':      profiel_vast,
            'profiel_slot':      profiel_slot,
            'ambulant_profiel':  ambulant_profiel_totaal,
            'config':            self.config,
        }
