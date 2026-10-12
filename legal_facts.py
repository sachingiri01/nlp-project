"""
legal_facts.py
Rule-based key-fact extraction for Hindi bail documents (no LLM, no ML).

Input : the *raw* facts text (before clean_text, so digits/punctuation are intact).
Output: a dict of facts + a helper to turn it into rows for display.
"""
import re

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NOT_DEV = r"(?![\u0900-\u097F])"          # next char must not be a Devanagari letter
_HI_SUFFIX = r"(?:बी|ए|सी|डी)" + _NOT_DEV
_EN_SUFFIX = r"[A-Da-d](?![A-Za-z])"
_SUFFIX_MAP = {"ए": "A", "बी": "B", "सी": "C", "डी": "D"}

# ---------------------------------------------------------------- acts
ACT_PATTERNS = {
    "IPC": r"भा[\W0]{0,2}द[\W0]{0,2}वि|आई[\W0]{0,2}पी[\W0]{0,2}सी|\bIPC\b|भारतीय\s*दण्ड\s*संहिता|भारतीय\s*दंड\s*संहिता",
    "NDPS": r"एन[\W0]{0,2}डी[\W0]{0,2}पी[\W0]{0,2}एस|\bNDPS\b|स्वापक|मादक\s*पदार्थ",
    "POCSO": r"पा?[ॉो]क्सो|पास्को|\bPOCSO\b",
    "Arms Act": r"आर्म्स\s*एक्ट|शस्त्र\s*अधिनियम|आयुध\s*अधिनियम",
    "Excise Act": r"आबकारी|एक्साइज",
    "Gangsters Act": r"गैंगस्टर|गैगस्टर",
}
_ACT_RE = {k: re.compile(v, re.I) for k, v in ACT_PATTERNS.items()}

# ---------------------------------------------------------------- sections
_SEC_BLOCK = re.compile(
    r"धारा(?:ओं|एं)?\s*((?:\d{1,3}\s*(?:" + _HI_SUFFIX + "|" + _EN_SUFFIX + r")?\s*"
    r"(?:[/,\-]|व" + _NOT_DEV + r"|और|एवं|तथा)?\s*)+)"
)
_SEC_ITEM = re.compile(r"(\d{1,3})\s*(" + _HI_SUFFIX + "|" + _EN_SUFFIX + ")?")
_SKIP_SECTIONS = {"34"}   # common-intention: carries no offence information

IPC_MAP = {
    "302": "Murder", "304": "Culpable homicide", "304B": "Dowry death",
    "307": "Attempt to murder", "308": "Attempt to commit culpable homicide",
    "376": "Sexual offence", "354": "Sexual offence", "354A": "Sexual offence",
    "363": "Kidnapping / abduction", "364": "Kidnapping / abduction",
    "366": "Kidnapping / abduction", "365": "Kidnapping / abduction",
    "379": "Theft", "380": "Theft", "411": "Theft / stolen property",
    "392": "Robbery / dacoity", "394": "Robbery / dacoity", "395": "Robbery / dacoity",
    "396": "Robbery / dacoity", "397": "Robbery / dacoity", "399": "Robbery / dacoity",
    "402": "Robbery / dacoity",
    "420": "Cheating / fraud", "406": "Breach of trust", "409": "Breach of trust",
    "467": "Forgery", "468": "Forgery", "471": "Forgery",
    "498A": "Cruelty / dowry harassment",
    "323": "Hurt", "324": "Hurt", "325": "Hurt", "326": "Hurt",
    "504": "Criminal intimidation", "506": "Criminal intimidation",
    "452": "House trespass", "147": "Rioting", "148": "Rioting", "149": "Rioting",
    "120B": "Criminal conspiracy",
}
NDPS_SECS = {"8", "15", "17", "18", "19", "20", "21", "22", "23", "27A", "29", "37", "50"}
POCSO_SECS = {"3", "4", "5", "6", "7", "8", "9", "10"}
ARMS_SECS = {"3", "25", "27"}
EXCISE_SECS = {"60", "63"}


def normalize_digits(text: str) -> str:
    return text.translate(_DEV_DIGITS)


def extract_sections(text: str):
    seen = []
    for block in _SEC_BLOCK.findall(text):
        for num, suf in _SEC_ITEM.findall(block):
            sec = num + _SUFFIX_MAP.get(suf, suf.upper())
            if sec not in _SKIP_SECTIONS and sec not in seen:
                seen.append(sec)
    return seen[:15]


def detect_acts(text: str):
    return [name for name, rx in _ACT_RE.items() if rx.search(text)]


def categorise(sections, acts):
    cats = []

    def add(c):
        if c not in cats:
            cats.append(c)

    for s in sections:
        if "NDPS" in acts and s in NDPS_SECS:
            add("Narcotics (NDPS)")
        elif "POCSO" in acts and s in POCSO_SECS:
            add("Sexual offence against child (POCSO)")
        elif "Arms Act" in acts and s in ARMS_SECS:
            add("Arms Act offence")
        elif "Excise Act" in acts and s in EXCISE_SECS:
            add("Excise / liquor offence")
        elif s in IPC_MAP:
            add(IPC_MAP[s])
    if "Gangsters Act" in acts:
        add("Gangsters Act")
    return cats


# ---------------------------------------------------------------- other facts
_DATE = r"\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}"
_CUSTODY_KW = r"(?:गिरफ्तार|निरुद्ध|जेल)"
_RX = {
    "crime_no": re.compile(r"(?:मु[\W0]{0,2}अ[\W0]{0,2}सं|मुकदमा\s*अपराध\s*संख्या|केस\s*क्राइम\s*नं)[\W0]{0,3}(\d+\s*/\s*\d{2,4})"),
    "station": re.compile(r"थाना\s*([^\s,।\-]+)"),
    "in_custody": re.compile(r"जेल\s*में\s*(?:निरुद्ध|बन्द|बंद)|न्यायिक\s*अभिरक्षा|निरुद्ध\s*(?:है|हैं|चला)"),
    "custody_before": re.compile("(" + _DATE + r")[^।]{0,80}?" + _CUSTODY_KW),
    "custody_after": re.compile(_CUSTODY_KW + r"[^।]{0,120}?(" + _DATE + ")"),
    "cs_not_filed": re.compile(r"आरोप[\s\-]*पत्र[^।]{0,40}?(?:नहीं|नही)|विवेचना\s*(?:लम्बित|लंबित|जारी|चल\s*रही)"),
    "cs_filed": re.compile(r"आरोप[\s\-]*पत्र[^।]{0,40}?(?:प्रेषित|दाखिल|प्रस्तुत|प्रेषण)|(?:प्रेषित|दाखिल)[^।]{0,20}आरोप[\s\-]*पत्र"),
    "no_history": re.compile(r"(?:कोई\s*)?(?:अन्य\s*)?आपराधिक\s*(?:इतिहास|अभिलेख)[^।]{0,20}?(?:नहीं|नही)|पूर्व\s*में\s*कोई[^।]{0,30}(?:नहीं|नही)"),
    "has_history": re.compile(r"हिस्ट्री\s*शीटर|हिस्ट्रीशीटर|आपराधिक\s*इतिहास[^।]{0,25}?(?:है|हैं|रहा|विवरण)"),
    "false_implication": re.compile(r"झूठ[ाे]?\s*(?:एवं\s*)?फंसा|फर्जी\s*(?:तरीके|तौर)|रंजिश|दुर्भावना"),
    "co_accused": re.compile(r"सह[\s\-]*अभियुक्त[^।]{0,100}?(?:जमानत|ज़मानत)[^।]{0,60}?(?:स्वीकार|मंजूर|रिहा)"),
    "minor": re.compile(r"नाबालिग|अवयस्क|किशोर"),
    "death": re.compile(r"मृतक|मृत्यु|हत्या"),
    "earlier_rejected": re.compile(r"जमानत[^।]{0,60}?(?:निरस्त|खारिज|अस्वीकार)"),
}


def _clean_date(d: str) -> str:
    return re.sub(r"\s+", "", d)


def extract_key_facts(text: str) -> dict:
    """Return a dict of extracted facts for one case (raw Hindi facts text)."""
    t = re.sub(r"\s+", " ", normalize_digits(text or ""))
    acts = detect_acts(t)
    secs = extract_sections(t)

    m = _RX["crime_no"].search(t)
    crime_no = m.group(1).replace(" ", "") if m else None
    m = _RX["station"].search(t)
    station = m.group(1) if m else None

    m = _RX["custody_before"].search(t) or _RX["custody_after"].search(t)
    custody_since = _clean_date(m.group(1)) if m else None
    in_custody = bool(_RX["in_custody"].search(t))

    if _RX["cs_not_filed"].search(t):
        chargesheet = "Not yet filed / investigation pending"
    elif _RX["cs_filed"].search(t):
        chargesheet = "Filed"
    else:
        chargesheet = None

    if _RX["no_history"].search(t):
        prior = "No prior criminal record stated"
    elif _RX["has_history"].search(t):
        prior = "Criminal history mentioned"
    else:
        prior = None

    return {
        "sections": secs,
        "acts": acts,
        "offence_categories": categorise(secs, acts),
        "crime_no": crime_no,
        "police_station": station,
        "in_custody": in_custody,
        "custody_since": custody_since,
        "chargesheet": chargesheet,
        "prior_record": prior,
        "false_implication_claimed": bool(_RX["false_implication"].search(t)),
        "co_accused_granted_bail": bool(_RX["co_accused"].search(t)),
        "minor_mentioned": bool(_RX["minor"].search(t)),
        "death_involved": bool(_RX["death"].search(t)),
        "earlier_bail_rejected": bool(_RX["earlier_rejected"].search(t)),
    }


def _yn(v):
    return "Yes" if v else "Not mentioned"


def facts_as_rows(f: dict):
    """(label, value) rows for the 'Extracted Key Facts' card."""
    return [
        ("Sections", ", ".join(f["sections"]) or "Not found"),
        ("Acts detected", ", ".join(f["acts"]) or "Not found"),
        ("Offence type", ", ".join(f["offence_categories"]) or "Not classified"),
        ("Case crime no.", f["crime_no"] or "Not found"),
        ("Police station", f["police_station"] or "Not found"),
        ("Accused in custody", "Yes" if f["in_custody"] else "Not mentioned"),
        ("Custody / arrest date", f["custody_since"] or "Not found"),
        ("Chargesheet", f["chargesheet"] or "Not mentioned"),
        ("Prior record", f["prior_record"] or "Not mentioned"),
        ("Claims false implication", _yn(f["false_implication_claimed"])),
        ("Co-accused already on bail", _yn(f["co_accused_granted_bail"])),
        ("Earlier bail rejected", _yn(f["earlier_bail_rejected"])),
        ("Death involved", _yn(f["death_involved"])),
        ("Minor mentioned", _yn(f["minor_mentioned"])),
    ]