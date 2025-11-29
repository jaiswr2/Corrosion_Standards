import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import streamlit as st
import re

# ============================================================
# GLOBAL CONFIG
# ============================================================
REL_TOL  = 0.20
BIGFONT  = 14

STANDARD_KEYS = [
    "Eurocode", "AS2159", "NZS", "WSDOT",
    "FDOT", "Japan", "China", "DIN", "Caltrans"
]

# FULL NAMES → SHOWN IN TABLE
DISPLAY_NAMES = {
    "Eurocode": "EN 1993-5:2007 (Eurocode)",
    "AS2159":   "AS 2159:2009 (Australia)",
    "NZS":      "NZS 3404-1:2009 (New Zealand)",
    "WSDOT":    "WSDOT BDM:2020 (USA)",
    "FDOT":     "FDOT SDG:2023 (USA)",
    "Japan":    "OCDI:2020 (Japan)",
    "China":    "JTG 3363:2019 (China)",
    "DIN":      "DIN 50929-3:2018 (Germany)",
    "Caltrans": "Caltrans BDM:2025 (USA)",
}

# SHORT NAMES → SHOWN IN BAR CHART
SHORT_NAMES = {
    "Eurocode": "EN",
    "AS2159": "AS",
    "NZS": "NZS",
    "WSDOT": "WSDOT",
    "FDOT": "FDOT",
    "Japan": "OCDI",
    "China": "JTG",
    "DIN": "DIN",
    "Caltrans": "Caltrans",
}

STANDARD_COLORS = {
    "Eurocode": "#000000",
    "AS2159":   "#0072B2",
    "NZS":      "#009E73",
    "WSDOT":    "#D55E00",
    "FDOT":     "#E69F00",
    "Japan":    "#CC79A7",
    "China":    "#8B4513",
    "DIN":      "#800080",
    "Caltrans": "#56B4E9",
}

FMT = FormatStrFormatter('%g')

# Common conceptual column names (for row-dicts)
COL_AGE     = "Age (yr)"
COL_PH      = "Soil_pH"
COL_CL      = "Chloride Content (mg/kg)"
COL_SO4     = "Sulphate_Content (mg/kg)"
COL_RHO     = "Soil_Resistivity (Ω·cm)"
COL_SOIL    = "Soil Type"
COL_LOC     = "Location wrt Water Table"
COL_FILL    = "Is_Fill_Material"
COL_FOREIGN = "Has_Foreign_Inclusions"
COL_FTYPE   = "Foreign_Inclusion_Type"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def to_num(x):
    if isinstance(x, str):
        x = x.replace(" ", "")
    return pd.to_numeric(x, errors="coerce")

def truthy01(v):
    """Convert Yes/No style or numeric to 0/1."""
    if pd.isna(v):
        return 0
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "yes", "true", "y"}:
            return 1
        try:
            return 1 if float(t) == 1 else 0
        except:
            return 0
    try:
        return 1 if float(v) == 1 else 0
    except:
        return 0

def clean_chloride(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    for bad in ["mg/kg", "mg", "kg", ","]:
        s = s.replace(bad, "")
    try:
        return float(s.strip())
    except:
        return np.nan

# ============================================================
# EUROCODE EN 1993-5
# ============================================================

EC_RATES = {
    "Non-compacted, aggressive fills":      0.057,
    "Non-compacted, non-aggressive fills":  0.022,
    "Aggressive natural soils":             0.032,
    "Polluted natural soils / industrial":  0.030,
    "Undisturbed natural soils":            0.012,
}

def eurocode_class(row):
    fill    = truthy01(row.get(COL_FILL))
    foreign = truthy01(row.get(COL_FOREIGN))
    ph  = to_num(row.get(COL_PH))
    cl  = to_num(row.get(COL_CL))
    so4 = to_num(row.get(COL_SO4))

    if fill == 1:
        return "Non-compacted, aggressive fills" if foreign == 1 \
               else "Non-compacted, non-aggressive fills"
    if pd.notna(ph) and ph < 6:
        return "Aggressive natural soils"
    if (pd.notna(cl) and cl > 300) or (pd.notna(so4) and so4 > 1000):
        return "Polluted natural soils / industrial"
    return "Undisturbed natural soils"

def predict_eurocode(row):
    cls = eurocode_class(row)
    return EC_RATES.get(cls, np.nan)

# ============================================================
# AS 2159:2009
# ============================================================

WT_FLUCT   = "fluctuation zone"
WT_IMMERS  = "permanent immersion"

AS_BANDS = {
    "Non-aggressive": (0.00, 0.01),
    "Mild":           (0.01, 0.02),
    "Moderate":       (0.02, 0.04),
    "Severe":         (0.04, 0.10),
    "Very Severe":    (0.10, float("inf")),
}

AS_RATES = {
    "Non-aggressive": 0.005,
    "Mild":           0.015,
    "Moderate":       0.030,
    "Severe":         0.070,
    "Very Severe":    0.100,
}

def first_token(soil):
    if not isinstance(soil, str) or not soil.strip():
        return None
    return re.split(r"[+/, \t-]+", soil)[0].upper()

def soil_is_coarse(first):
    coarse = {"GW","GP","SW","SP","SM","GM","S","G","SC"}
    return first in coarse

def loc_is_below_WT(txt):
    if not isinstance(txt,str):
        return False
    t = txt.lower().strip()
    return (t == WT_FLUCT) or (t == WT_IMMERS)

def pH_bin(ph):
    if pd.isna(ph): return "pH_unk"
    if ph < 4:     return "pH_<4"
    if ph < 5:     return "pH_4_5"
    if ph < 8.5:   return "pH_5_8p5"
    return "pH_>=8p5"

def Cl_bin(cl):
    if pd.isna(cl): return "Cl_unk"
    if cl < 5000:     return "Cl_<5k"
    if cl < 20000:    return "Cl_5k_20k"
    return "Cl_>=20k"

def upgrade_one(level):
    order=["Non-aggressive","Mild","Moderate","Severe","Very Severe"]
    try:
        i=order.index(level)
        return order[min(i+1,len(order)-1)]
    except:
        return level

BASE_COND_A = {
    ("pH_<4","Cl_<5k"):"Moderate",
    ("pH_<4","Cl_5k_20k"):"Severe",
    ("pH_<4","Cl_>=20k"):"Very Severe",

    ("pH_4_5","Cl_<5k"):"Mild",
    ("pH_4_5","Cl_5k_20k"):"Moderate",
    ("pH_4_5","Cl_>=20k"):"Severe",

    ("pH_5_8p5","Cl_<5k"):"Non-aggressive",
    ("pH_5_8p5","Cl_5k_20k"):"Mild",
    ("pH_5_8p5","Cl_>=20k"):"Moderate",

    ("pH_>=8p5","Cl_<5k"):"Non-aggressive",
    ("pH_>=8p5","Cl_5k_20k"):"Mild",
    ("pH_>=8p5","Cl_>=20k"):"Moderate",
}

BASE_COND_B = {
    ("pH_<4","Cl_<5k"):"Moderate",
    ("pH_<4","Cl_5k_20k"):"Severe",
    ("pH_<4","Cl_>=20k"):"Very Severe",

    ("pH_4_5","Cl_<5k"):"Non-aggressive",
    ("pH_4_5","Cl_5k_20k"):"Non-aggressive",
    ("pH_4_5","Cl_>=20k"):"Mild",

    ("pH_5_8p5","Cl_<5k"):"Non-aggressive",
    ("pH_5_8p5","Cl_5k_20k"):"Mild",
    ("pH_5_8p5","Cl_>=20k"):"Moderate",

    ("pH_>=8p5","Cl_<5k"):"Non-aggressive",
    ("pH_>=8p5","Cl_5k_20k"):"Mild",
    ("pH_>=8p5","Cl_>=20k"):"Moderate",
}

def classify_as2159_row(row):
    # foreign inclusion → very severe
    if truthy01(row.get(COL_FOREIGN)) == 1:
        return "Very Severe"
    fi = str(row.get(COL_FTYPE,"")).lower()
    if any(k in fi for k in ["cinder","fly","wood"]):
        return "Very Severe"

    first = first_token(row.get(COL_SOIL))
    belowWT = loc_is_below_WT(str(row.get(COL_LOC,"")))
    cond = "A" if (belowWT or soil_is_coarse(first)) else "B"

    phb = pH_bin(to_num(row.get(COL_PH)))
    clb = Cl_bin(to_num(row.get(COL_CL)))

    exposure = (BASE_COND_A if cond=="A" else BASE_COND_B).get((phb,clb),"Non-aggressive")

    so4 = to_num(row.get(COL_SO4))
    if pd.notna(so4) and so4 > 1000:
        exposure = upgrade_one(exposure)

    return exposure

def predict_as2159(row):
    cls = classify_as2159_row(row)
    return AS_RATES.get(cls, np.nan)

# ============================================================
# NZS 3404.1:2009
# ============================================================

NZS_RATES = {
    "Buried in fill below WT":                          0.015,
    "Buried in controlled fill above WT":               0.015,
    "Buried in uncontrolled fill above WT (pH≥4)":      0.050,
    "Buried in uncontrolled fill above WT (pH<4)":      0.075,
    "Buried in rubble fill (concrete/brick/inorganic)": 0.025,
    "Undisturbed natural soil":                         0.015,
}

def is_below_WT(txt):
    if not isinstance(txt,str): return False
    t=txt.lower()
    return ("below" in t) or ("immersion" in t) or ("fluctuation" in t)

def is_rubble(txt):
    if not isinstance(txt,str): return False
    t=txt.lower()
    return ("concrete" in t) or ("brick" in t) or ("rubble" in t)

def classify_nzs_row(row):
    ph = to_num(row.get(COL_PH))
    fill    = truthy01(row.get(COL_FILL))
    foreign = truthy01(row.get(COL_FOREIGN))
    loc     = row.get(COL_LOC,"")
    soil    = row.get(COL_SOIL,"")

    if fill==1:
        if is_below_WT(loc):
            cls="Buried in fill below WT"
            return cls, NZS_RATES[cls]
        if foreign==1:
            if pd.notna(ph) and ph<4:
                cls="Buried in uncontrolled fill above WT (pH<4)"
            else:
                cls="Buried in uncontrolled fill above WT (pH≥4)"
            return cls, NZS_RATES[cls]
        if is_rubble(soil):
            cls="Buried in rubble fill (concrete/brick/inorganic)"
            return cls, NZS_RATES[cls]
        cls="Buried in controlled fill above WT"
        return cls, NZS_RATES[cls]

    cls="Undisturbed natural soil"
    return cls, NZS_RATES[cls]

def predict_nzs(row):
    _, rate = classify_nzs_row(row)
    return rate

# ============================================================
# WSDOT BDM 2020
# ============================================================

WSDOT_RATES = {
    "Undisturbed (non-corrosive)"       : 0.01270,
    "Undisturbed (corrosive)"           : 0.02540,
    "Fill/disturbed (non-corrosive)"    : 0.01905,
    "Fill/disturbed (corrosive)"        : 0.03810,
}

def is_corrosive_ws(ph, cl, so4):
    if pd.notna(cl) and cl >= 500: return True
    if pd.notna(so4) and so4 >= 1500: return True
    if pd.notna(ph) and ph <= 5.5: return True
    return False

def classify_ws(row):
    ph  = to_num(row.get(COL_PH))
    cl  = to_num(row.get(COL_CL))
    so4 = to_num(row.get(COL_SO4))
    disturbed = truthy01(row.get(COL_FILL))
    corr = is_corrosive_ws(ph, cl, so4)

    if disturbed == 1:
        name = "Fill/disturbed (corrosive)" if corr else "Fill/disturbed (non-corrosive)"
    else:
        name = "Undisturbed (corrosive)" if corr else "Undisturbed (non-corrosive)"
    return name, WSDOT_RATES[name]

def predict_wsdot(row):
    _, rate = classify_ws(row)
    return rate

# ============================================================
# FDOT 2023
# ============================================================

FDOT_RATES = {
    ("Slightly aggressive",   "Partially buried"):  0.038,
    ("Slightly aggressive",   "Completely buried"): 0.025,
    ("Moderately aggressive", "Partially buried"):  0.051,
    ("Moderately aggressive", "Completely buried"): 0.038,
    ("Extremely aggressive",  "Partially buried"):  0.064,
    ("Extremely aggressive",  "Completely buried"): 0.051,
}

def burial_from_location(txt):
    if isinstance(txt,str) and ("fluctuation" in txt.lower()):
        return "Partially buried"
    return "Completely buried"

def fdot_aggr(ph, rho, cl):
    extreme = False
    if pd.notna(ph) and ph < 6: extreme = True
    if pd.notna(rho) and rho < 1000: extreme = True
    if pd.notna(cl) and cl >= 2000: extreme = True
    if extreme: return "Extremely aggressive"

    moderate = False
    if pd.notna(ph) and 6 <= ph <= 7: moderate = True
    if pd.notna(rho) and 1000 <= rho <= 5000: moderate = True
    if pd.notna(cl) and 500 <= cl < 2000: moderate = True
    if moderate: return "Moderately aggressive"

    return "Slightly aggressive"

def predict_fdot(row):
    ph  = to_num(row.get(COL_PH))
    cl  = to_num(row.get(COL_CL))
    rho = to_num(row.get(COL_RHO))
    ag  = fdot_aggr(ph, rho, cl)
    buri = burial_from_location(row.get(COL_LOC))
    return FDOT_RATES.get((ag,buri), np.nan)

# ============================================================
# Japan (OCDI 2020)
# ============================================================

JAPAN_RATES = {
    "Above residual water level": 0.030,
    "Below residual water level": 0.020,
}

def classify_japan_location(txt):
    if isinstance(txt, str):
        t = txt.lower()
        if ("above" in t) or ("fluctuation" in t):
            return "Above residual water level", JAPAN_RATES["Above residual water level"]
        if ("below" in t) or ("permanent" in t):
            return "Below residual water level", JAPAN_RATES["Below residual water level"]
    return "Below residual water level", JAPAN_RATES["Below residual water level"]

def predict_japan(row):
    _, rate = classify_japan_location(row.get(COL_LOC, ""))
    return rate

# ============================================================
# China JTG 3363-2019
# ============================================================

CHINA_RATES = {
    "Above / Fluctuation": 0.060,
    "Below":               0.030,
}

def classify_china_location(txt):
    if not isinstance(txt, str) or txt.strip() == "":
        return "Above / Fluctuation", CHINA_RATES["Above / Fluctuation"]
    t = txt.lower()
    if ("permanent" in t) or ("immersion" in t):
        return "Below", CHINA_RATES["Below"]
    return "Above / Fluctuation", CHINA_RATES["Above / Fluctuation"]

def predict_china(row):
    _, rate = classify_china_location(row.get(COL_LOC, ""))
    return rate

# ============================================================
# Caltrans BDM 2021
# ============================================================

CALTRANS_RATES = {
    "Natural Soil":          0.025,
    "Fill/Disturbed":        0.0381,
    "Highly Corrosive Fill": 0.05,
    "Not Corrosive":         0.001,
}

def caltrans_classification(row):
    foreign = row.get(COL_FOREIGN, 0)
    fill    = row.get(COL_FILL, 0)

    foreign = int(foreign) if pd.notna(foreign) else 0
    fill    = int(fill)    if pd.notna(fill)    else 0

    ph  = to_num(row.get(COL_PH))
    cl  = to_num(row.get(COL_CL))
    so4 = to_num(row.get(COL_SO4))

    if foreign == 1:
        return "Highly Corrosive Fill", CALTRANS_RATES["Highly Corrosive Fill"]

    corrosive = False
    if (pd.notna(ph) and ph <= 5.5) or (pd.notna(cl) and cl >= 500) or (pd.notna(so4) and so4 >= 1500):
        corrosive = True

    if corrosive:
        if fill == 1:
            return "Fill/Disturbed", CALTRANS_RATES["Fill/Disturbed"]
        else:
            return "Natural Soil", CALTRANS_RATES["Natural Soil"]
    else:
        return "Not Corrosive", CALTRANS_RATES["Not Corrosive"]

def predict_caltrans(row):
    _, rate = caltrans_classification(row)
    return rate

# ============================================================
# DIN 50929-3:2018 – UNIFORM CORROSION RATE ONLY
# ============================================================

FOREIGN_KEYWORDS = [
    r"\bfly\s*ash\b", r"\bflyash\b", r"\bash(es)?\b",
    r"\bwood\b", r"\bshred+ed?\s*wood\b", r"\bshredd?ed\b",
    r"\brubble\b", r"\bslag\b", r"\bpeat\b", r"\bfen\b",
    r"\bmud\b", r"\bmarsh\b", r"\brefuse\b", r"\bwaste\b"
]

def has_foreign_inclusion_from_text(soil_type_text):
    if not isinstance(soil_type_text, str) or not soil_type_text.strip():
        return False
    s = soil_type_text.lower()
    for pat in FOREIGN_KEYWORDS:
        if re.search(pat, s):
            return True
    if any(sep in s for sep in "+/,;"):
        parts = re.split(r"[+/;,]", s)
        return any(has_foreign_inclusion_from_text(p) for p in parts)
    return False

def first_uscs_symbol(text):
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.upper().replace("\\", "/")
    for sep in ("+", "/", ",", ";", " "):
        if sep in s:
            s = s.split(sep)[0]
            break
    return s.strip()

def z1_from_uscs_autoflag(soil_type_text, explicit_flag=None):
    base = 0
    sym = first_uscs_symbol(soil_type_text)
    if sym:
        if sym in {"GW","GP","SW","SP"}: base = +4
        elif "-" in sym:                 base = +2
        elif sym in {"GM","GC","SM","SC"}: base = +2
        elif sym in {"ML","CL","MH","CH","OL","OH"}: base = -2
        elif sym == "PT":                base = -2
    contam = bool(explicit_flag) or has_foreign_inclusion_from_text(soil_type_text)
    return base + (-12 if contam else 0)

def z2_from_resistivity_ohm_cm(r):
    if r is None or (isinstance(r, float) and pd.isna(r)): return 0
    r = float(r)
    if r > 50000: return +4
    if r > 20000: return +2
    if r >  5000: return 0
    if r >  2000: return -2
    if r >  1000: return -4
    return -6

def z3_from_moisture(m):
    # For GUI we do not have moisture → assume <=20% → Z3=0
    if m is None or (isinstance(m, float) and pd.isna(m)): return 0
    return -1 if float(m) > 20.0 else 0

def z4_from_ph(p):
    if p is None or (isinstance(p, float) and pd.isna(p)): return 0
    p = float(p)
    if p > 9.0: return +2
    if p >= 6.0: return 0
    if p >= 4.0: return -1
    return -3

def mmol_per_kg_from_mg_per_kg(mg_per_kg, mm):
    if mg_per_kg is None or (isinstance(mg_per_kg, float) and pd.isna(mg_per_kg)): return 0.0
    return float(mg_per_kg)/mm

def z8_from_sulfate_mgkg_acid_extract(so4_mgkg):
    so4 = mmol_per_kg_from_mg_per_kg(so4_mgkg, 96.06)
    if so4 < 2:   return 0
    if so4 <= 5:  return -1
    if so4 <= 10: return -2
    return -3

def z9_from_neutral_salts(cl_mgkg, so4_mgkg):
    c_cl  = mmol_per_kg_from_mg_per_kg(cl_mgkg, 35.45)
    c_so4 = mmol_per_kg_from_mg_per_kg(so4_mgkg, 96.06)
    c_ns = c_cl + 2.0*c_so4
    if c_ns < 3:    return 0
    if c_ns <= 10:  return -1
    if c_ns <= 30:  return -2
    if c_ns <= 100: return -3
    return -4

def z10_from_water_table(s):
    if not isinstance(s, str): return 0
    s = s.strip().lower()
    if "fluct" in s: return -2
    if "perm" in s or "immer" in s or "below" in s: return -1
    return 0

def z13_from_flags(explicit_flag, soil_text):
    return -6 if (bool(explicit_flag) or has_foreign_inclusion_from_text(soil_text)) else 0

def din_rate_bin_from_total(total):
    if total >= 0:           return 0.005, 0.03
    if -4 <= total <= -1:    return 0.01,  0.05
    if -10 <= total <= -5:   return 0.02,  0.20
    return 0.06, 0.40

def predict_din_uniform(row):
    soil_text = row.get(COL_SOIL)
    explicit_contam = truthy01(row.get(COL_FOREIGN))
    ph  = to_num(row.get(COL_PH))
    cl  = to_num(row.get(COL_CL))
    rho = to_num(row.get(COL_RHO))
    so4 = to_num(row.get(COL_SO4))
    loc = row.get(COL_LOC)

    # Moisture not in GUI → assume None → Z3=0
    moisture = None

    z1 = z1_from_uscs_autoflag(soil_text, explicit_flag=explicit_contam)
    z2 = z2_from_resistivity_ohm_cm(rho)
    z3 = z3_from_moisture(moisture)
    z4 = z4_from_ph(ph)
    z5 = 0
    z6 = 0
    z8 = z8_from_sulfate_mgkg_acid_extract(so4)
    z9 = z9_from_neutral_salts(cl, so4)
    z10 = z10_from_water_table(loc)
    z11 = 0
    z12 = 0
    z13 = z13_from_flags(explicit_contam, soil_text)
    z14 = 0

    b0 = z1 + z2 + z3 + z4 + z5 + z6 + z8 + z9 + z10
    b1 = b0 + z11 + z12 + z13 + z14
    w, wL = din_rate_bin_from_total(b1)
    return w  # uniform corrosion rate (mm/yr)

# ============================================================
# WRAPPER: COMPUTE ALL STANDARDS
# ============================================================

def compute_all_standards(row):
    """
    row: dict with entries for COL_AGE, COL_PH, COL_CL, COL_SO4,
         COL_RHO, COL_SOIL, COL_LOC, COL_FILL, COL_FOREIGN, COL_FTYPE
    Returns: dict {standard_key: rate_mm_per_yr}
    """
    rates = {}

    rates["Eurocode"] = predict_eurocode(row)
    rates["AS2159"]   = predict_as2159(row)
    rates["NZS"]      = predict_nzs(row)
    rates["WSDOT"]    = predict_wsdot(row)
    rates["FDOT"]     = predict_fdot(row)
    rates["Japan"]    = predict_japan(row)
    rates["China"]    = predict_china(row)
    rates["Caltrans"] = predict_caltrans(row)
    rates["DIN"]      = predict_din_uniform(row)

    return rates

# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Steel Pile Corrosion – Design Standards",
        layout="centered"
    )

    st.title("Estimation of Corrosion of Steel Piles in Soil (Design Standards)")

    st.markdown(
        """
        This tool estimates **uniform corrosion rate (mm/yr)** and corresponding  
        **thickness loss (mm)** for steel piles based on multiple international standards.
        """
    )

    st.subheader("Input Soil / Environment Parameters")

    # ============================================================
    # EXACT 5-ROW INPUT LAYOUT
    # ============================================================

    col1, col2 = st.columns(2)

    # ------- ROW 1 -------
    with col1:
        age = st.number_input("Age (years)", min_value=1.0, max_value=200.0,
                              value=34.0, step=1.0)
    with col2:
        ph = st.number_input("Soil pH", min_value=2.0, max_value=12.0,
                             value=7.8, step=0.1)

    # ------- ROW 2 -------
    with col1:
        cl = st.number_input("Chloride (mg/kg)", min_value=0.0, max_value=50000.0,
                             value=444.0, step=10.0)
    with col2:
        so4 = st.number_input("Sulphate (mg/kg)", min_value=0.0, max_value=50000.0,
                              value=328.0, step=10.0)

    # ------- ROW 3 -------
    with col1:
        rho = st.number_input("Soil Resistivity (Ω·cm)", min_value=50.0, max_value=50000.0,
                              value=900.0, step=50.0)
    with col2:
        soil_type = st.selectbox("Soil Type (USCS)",
                                 ["CL","SM","ML","SP","CH","GP","SW","OL","SC","GT","GW"])

    # ------- ROW 4 -------
    with col1:
        loc = st.selectbox("Location wrt Water Table",
                           ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"])
    with col2:
        fill_flag = 1 if st.selectbox("Is Fill Material?", ["No", "Yes"]) == "Yes" else 0

    # ------- ROW 5 -------
    with col1:
        foreign_flag = 1 if st.selectbox("Has Foreign Inclusions (rubble, wood, etc.)?",
                                         ["No", "Yes"]) == "Yes" else 0
    with col2:
        foreign_type = st.selectbox("Foreign Inclusion Type",
                                    ["None", "Flyash", "Shredded wood", "Cinder"])

    # Build row dict
    row = {
        COL_AGE:     age,
        COL_PH:      ph,
        COL_CL:      cl,
        COL_SO4:     so4,
        COL_RHO:     rho,
        COL_SOIL:    soil_type,
        COL_LOC:     loc,
        COL_FILL:    fill_flag,
        COL_FOREIGN: foreign_flag,
        COL_FTYPE:   foreign_type,
    }

    st.markdown("---")

    # ============================================================
    # PROCESS INPUT
    # ============================================================
    if st.button("Estimate Corrosion Rates"):
        rates = compute_all_standards(row)

        # ============================================================
        # TABLE OUTPUT (RATE + LOSS)
        # ============================================================
        data = []
        for key in STANDARD_KEYS:
            rate = rates.get(key, np.nan)
            loss = rate * age
            data.append({
                "Design Standard": DISPLAY_NAMES.get(key, key),
                "Corrosion Rate (mm/yr)": rate,
                "Thickness Loss over years (mm)": loss
            })

        df_out = pd.DataFrame(data)

        st.subheader("Estimated Corrosion Rate & Thickness Loss")
        st.dataframe(
            df_out.style.format({
                "Corrosion Rate (mm/yr)": "{:.4f}",
                "Thickness Loss (mm)": "{:.3f}"
            }),
            use_container_width=True
        )

        # ============================================================
        # BAR CHART
        # ============================================================
        st.subheader("Comparison of Corrosion Rate (mm/yr)")

        fig, ax = plt.subplots(figsize=(11, 6))

        x_labels = [SHORT_NAMES[k] for k in STANDARD_KEYS]
        y_vals = [rates[k] for k in STANDARD_KEYS]
        x_pos = np.arange(len(x_labels))
        colors = [STANDARD_COLORS.get(k, "gray") for k in STANDARD_KEYS]

        ax.bar(x_pos, y_vals, color=colors, edgecolor="black")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=14)  # larger font
        ax.set_ylabel("Corrosion Rate (mm/yr)", fontsize=16)

        # Y-tick font size increase
        plt.yticks(fontsize=16)

        # Value labels: bigger font
        for i, v in enumerate(y_vals):
            ax.text(i, v + max(y_vals)*0.02,
                    f"{v:.4f}",
                    ha="center", va="bottom",
                    fontsize=14)

        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
