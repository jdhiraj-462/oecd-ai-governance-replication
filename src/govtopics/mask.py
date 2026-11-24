"""Mask geographic and national identifiers before topic modelling.

Without masking, BERTopic groups initiatives by who wrote them (topics such as
"republic serbia" or "saudi arabia") rather than by what they govern, which makes
country comparisons on those topics circular. Place names, nationalities,
languages and currency codes are replaced by a neutral token. Masking uses spaCy
NER (en_core_web_sm: GPE, NORP, LOC, LANGUAGE) plus an exact list of country names
(pycountry and the jurisdictions in the corpus). Agency names and acronyms are not
masked; the sensitivity analysis reports results without masking.
"""
from __future__ import annotations

import re

TOKEN = "PLACE"
LABELS = {"GPE", "NORP", "LOC", "LANGUAGE"}
CURRENCY = re.compile(r"\b(AUD|CAD|SGD|JPY|USD|EUR|GBP|INR|NOK|SEK|DKK|CHF|KRW|NZD|BRL|CNY|RMB|HKD|ZAR)\b")
EXTRA = ["European Union", "EU", "Europe", "European", "UK", "U.S.", "US", "USA", "UAE", "Korea", "Türkiye",
         "Flanders", "Flemish", "Wallonia", "Walloon", "Nordic", "Baltic", "Nordic-Baltic", "Latin America",
         "Africa", "African", "Asia", "Asian", "ASEAN", "Gulf", "China", "Chinese", "Czech", "Slovak",
         "Vietnam", "Ivorian", "Turkey", "Russia", "Russian"]
# Demonyms and language adjectives for the jurisdictions in the corpus
DEMONYMS = """Albanian Algerian Argentine Argentinian Armenian Australian Austrian Azerbaijani Belgian Beninese
Brazilian Bruneian Bulgarian Cambodian Khmer Cameroonian Canadian Chilean Colombian Costa-Rican Croatian Cuban
Cypriot Danish Dane Dominican Ecuadorian Egyptian Estonian Ethiopian Amharic Finnish Finn French German Ghanaian
Greek Hungarian Icelandic Indian Hindi Indonesian Irish Israeli Hebrew Italian Japanese Kazakh Kazakhstani Kenyan
Swahili Korean Latvian Basotho Libyan Lithuanian Luxembourgish Luxembourger Malaysian Malay Maltese Mauritanian
Mauritian Mexican Moroccan Dutch Nigerien Nigerian Norwegian Pakistani Urdu Peruvian Filipino Philippine Polish
Portuguese Romanian Rwandan Saudi Senegalese Serbian Singaporean Slovak Slovakian Slovenian Slovene Spanish
Castilian Catalan Basque Galician Swedish Swiss Thai Tunisian Turkish Ugandan Ukrainian Emirati British English
Scottish Welsh American Uruguayan Uzbek Vietnamese Zambian Zimbabwean Arab Arabic Latin-American Iberian
Scandinavian Balkan Brussels Paris Berlin London Madrid Lisbon Riyadh Tokyo Seoul Delhi Ottawa Canberra""".split()
PROTECT = {"AI", "A.I.", "GenAI", "AGI", "ICT", "IoT", "SME", "SMEs", "R&D", "HPC", "GPAI"}


def _names(extra_names) -> list[str]:
    names = set(EXTRA) | {d.replace("-", " ") for d in DEMONYMS} | {n for n in extra_names if isinstance(n, str) and len(n) > 2}
    try:
        import pycountry
        for c in pycountry.countries:
            for a in ("name", "common_name", "official_name"):

                if getattr(c, a, None):
                    names.add(getattr(c, a))
    except ImportError:
        pass
    return sorted(names, key=len, reverse=True)


def mask_texts(texts: list[str], extra_names=()) -> tuple[list[str], dict]:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in _names(extra_names)) + r")(?:'s|’s|s)?\b")
    out, n_ent, n_list = [], 0, 0
    for doc in nlp.pipe(texts, batch_size=64):
        s, last, parts = doc.text, 0, []
        for e in doc.ents:
            if e.label_ in LABELS and e.text.strip() not in PROTECT and not e.text.strip().upper().startswith("AI "):
                parts.append(s[last:e.start_char]); parts.append(TOKEN); last = e.end_char; n_ent += 1
        parts.append(s[last:])
        t = "".join(parts)
        t, k = pat.subn(TOKEN, t); n_list += k
        t = CURRENCY.sub(TOKEN, t)
        t = re.sub(rf"(?:{TOKEN}[\s,/-]*(?:and|or|of|the)?\s*)+{TOKEN}", TOKEN, t)  # collapse runs
        out.append(t)
    return out, {"entities_masked": n_ent, "list_matches_masked": n_list}
