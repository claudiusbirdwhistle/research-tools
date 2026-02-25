"""Currency definitions for the contagion analysis."""

CURRENCIES = {
    "BRL": {"name": "Brazilian Real", "country": "Brazil", "region": "LatAm", "type": "EM"},
    "MXN": {"name": "Mexican Peso", "country": "Mexico", "region": "LatAm", "type": "EM"},
    "ZAR": {"name": "South African Rand", "country": "South Africa", "region": "Africa", "type": "EM"},
    "TRY": {"name": "Turkish Lira", "country": "Turkey", "region": "EMEA", "type": "EM"},
    "PLN": {"name": "Polish Zloty", "country": "Poland", "region": "CEE", "type": "EM"},
    "HUF": {"name": "Hungarian Forint", "country": "Hungary", "region": "CEE", "type": "EM"},
    "CZK": {"name": "Czech Koruna", "country": "Czech Republic", "region": "CEE", "type": "EM"},
    "KRW": {"name": "South Korean Won", "country": "South Korea", "region": "Asia", "type": "EM"},
    "THB": {"name": "Thai Baht", "country": "Thailand", "region": "Asia", "type": "EM"},
    "INR": {"name": "Indian Rupee", "country": "India", "region": "Asia", "type": "EM"},
    "IDR": {"name": "Indonesian Rupiah", "country": "Indonesia", "region": "Asia", "type": "EM"},
    "PHP": {"name": "Philippine Peso", "country": "Philippines", "region": "Asia", "type": "EM"},
    "GBP": {"name": "British Pound", "country": "UK", "region": "Europe", "type": "DM"},
    "JPY": {"name": "Japanese Yen", "country": "Japan", "region": "Asia", "type": "DM"},
    "CHF": {"name": "Swiss Franc", "country": "Switzerland", "region": "Europe", "type": "DM"},
    "AUD": {"name": "Australian Dollar", "country": "Australia", "region": "Oceania", "type": "DM"},
    "CAD": {"name": "Canadian Dollar", "country": "Canada", "region": "N. America", "type": "DM"},
    "SEK": {"name": "Swedish Krona", "country": "Sweden", "region": "Europe", "type": "DM"},
    "NOK": {"name": "Norwegian Krone", "country": "Norway", "region": "Europe", "type": "DM"},
    "MYR": {"name": "Malaysian Ringgit", "country": "Malaysia", "region": "Asia", "type": "EM"},
}

EM_CURRENCIES = [c for c, d in CURRENCIES.items() if d["type"] == "EM"]
DM_CURRENCIES = [c for c, d in CURRENCIES.items() if d["type"] == "DM"]
ALL_CODES = sorted(CURRENCIES.keys())

KNOWN_CRISES = [
    {"name": "Brazilian/Argentine Crisis", "start": "1999-01-01", "end": "2002-06-30",
     "epicenter": ["BRL"], "description": "BRL devaluation Jan 1999, Argentina default Dec 2001"},
    {"name": "Global Financial Crisis", "start": "2008-07-01", "end": "2009-06-30",
     "epicenter": ["GBP", "KRW", "HUF"], "description": "Lehman Sep 2008, global EM sell-off"},
    {"name": "European Debt Crisis", "start": "2010-04-01", "end": "2012-07-31",
     "epicenter": ["HUF", "PLN", "CZK"], "description": "Greece May 2010, contagion to PIIGS"},
    {"name": "Taper Tantrum", "start": "2013-05-01", "end": "2013-09-30",
     "epicenter": ["BRL", "INR", "IDR", "ZAR", "TRY"], "description": "Fed taper signal, Fragile Five sell-off"},
    {"name": "Commodity/China Crash", "start": "2014-07-01", "end": "2016-02-28",
     "epicenter": ["BRL", "ZAR", "MYR", "NOK"], "description": "Oil crash, CNY devaluation Aug 2015"},
    {"name": "EM Crisis 2018", "start": "2018-04-01", "end": "2018-11-30",
     "epicenter": ["TRY", "ZAR", "BRL"], "description": "Turkey/Argentina Aug 2018 contagion"},
    {"name": "COVID Crash", "start": "2020-02-15", "end": "2020-06-30",
     "epicenter": ["BRL", "ZAR", "MXN"], "description": "Global risk-off Feb-Mar 2020"},
    {"name": "Russia/Rate Shock", "start": "2022-02-01", "end": "2022-10-31",
     "epicenter": ["HUF", "PLN", "CZK", "TRY"], "description": "Ukraine invasion, global rate tightening"},
]
