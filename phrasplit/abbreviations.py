"""Language-dependent abbreviations for sentence segmentation.

This module contains abbreviations that should not trigger sentence splits
for various languages. These are used by the post-processing corrections
in the splitter module.

Supported languages (based on spaCy models):
- Catalan (ca)
- Chinese (zh) - no abbreviations needed
- Croatian (hr)
- Danish (da)
- Dutch (nl)
- English (en)
- Finnish (fi)
- French (fr)
- German (de)
- Greek (el)
- Italian (it)
- Japanese (ja) - no abbreviations needed
- Korean (ko) - no abbreviations needed
- Lithuanian (lt)
- Macedonian (mk)
- Norwegian Bokmal (nb)
- Polish (pl)
- Portuguese (pt)
- Romanian (ro)
- Russian (ru)
- Slovenian (sl)
- Spanish (es)
- Swedish (sv)
- Ukrainian (uk)
"""

from __future__ import annotations

import re

from phrasplit.spacy_models import normalize_spacy_language

# Single letters used in names (language-independent, Latin alphabet)
_SINGLE_LETTERS_LATIN: set[str] = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
}

# Single letters for Cyrillic alphabet
_SINGLE_LETTERS_CYRILLIC: set[str] = {
    "А",
    "Б",
    "В",
    "Г",
    "Д",
    "Е",
    "Ж",
    "З",
    "И",
    "Й",
    "К",
    "Л",
    "М",
    "Н",
    "О",
    "П",
    "Р",
    "С",
    "Т",
    "У",
    "Ф",
    "Х",
    "Ц",
    "Ч",
    "Ш",
    "Щ",
    "Ы",
    "Э",
    "Ю",
    "Я",
}

# Single letters for Greek alphabet
_SINGLE_LETTERS_GREEK: set[str] = {
    "Α",
    "Β",
    "Γ",
    "Δ",
    "Ε",
    "Ζ",
    "Η",
    "Θ",
    "Ι",
    "Κ",
    "Λ",
    "Μ",
    "Ν",
    "Ξ",
    "Ο",
    "Π",
    "Ρ",
    "Σ",
    "Τ",
    "Υ",
    "Φ",
    "Χ",
    "Ψ",
    "Ω",
}

# =============================================================================
# English abbreviations
# =============================================================================
ENGLISH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Prof",
    "Rev",
    "Sr",
    "Jr",
    "St",
    "Gen",
    "Col",
    "Lt",
    "Capt",
    "Sgt",
    "Rep",
    "Sen",
    "Gov",
    "Pres",
    "Hon",
    # Academic degrees
    "Ph",  # Ph.D.
    "B",  # B.A., B.S., B.Sc.
    "D",  # D.D., D.Phil.
    # Common abbreviations
    "Inc",
    "Corp",
    "Ltd",
    "Co",
    "vs",
    "etc",
    "al",  # et al.
    "approx",
    "dept",
    "est",
    "vol",
    "nos",
    "fig",
    "figs",
    # Geographic
    "Mt",
    "Ft",
    "Ave",
    "Blvd",
    "Rd",
    # Units (when abbreviated with period)
    "oz",
    "lb",
    "ft",
    "in",
    "yr",
    "mo",
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# German abbreviations
# =============================================================================
GERMAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Dr",
    "Prof",
    "Hr",  # Herr
    "Fr",  # Frau
    "Dipl",  # Diplom
    "Ing",  # Ingenieur
    "Med",  # Medizin
    "rer",  # rerum (Dr. rer. nat.)
    "nat",  # naturarum
    "phil",  # philosophiae
    "jur",  # juris
    # Common abbreviations
    "bzw",  # beziehungsweise
    "ca",  # circa
    "etc",
    "evtl",  # eventuell
    "ggf",  # gegebenenfalls
    "inkl",  # inklusive
    "max",
    "min",
    "Nr",  # Nummer
    "sog",  # sogenannt
    "usw",  # und so weiter
    "vgl",  # vergleiche
    "z",  # z.B.
    "u",  # u.a.
    "o",  # o.ä.
    "d",  # d.h.
    "h",  # d.h.
    "a",  # u.a., o.ä.
    "ä",  # o.ä.
    "Ä",  # o.Ä.
    "s",  # s.o., s.u.
    # Geographic
    "Str",  # Straße
    # Company types
    "GmbH",
    "AG",
    "KG",
    "Co",
    "e",  # e.V.
    "V",  # e.V.
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# French abbreviations
# =============================================================================
FRENCH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Mme",  # Madame
    "Mlle",  # Mademoiselle
    "Dr",
    "Prof",
    "Me",  # Maître
    "Mgr",  # Monseigneur
    # Common abbreviations
    "etc",
    "cf",  # confer
    "env",  # environ
    "max",
    "min",
    "vol",
    "no",  # numéro
    "p",  # page
    "pp",  # pages
    "ex",  # exemple
    "c",  # c.-à-d.
    "à",  # c.-à-d.
    "d",  # c.-à-d.
    "av",  # avant
    "apr",  # après
    # Geographic
    "bd",  # boulevard
    "pl",  # place
    # Company types
    "SA",
    "Cie",
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Spanish abbreviations
# =============================================================================
SPANISH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Sr",  # Señor
    "Sra",  # Señora
    "Srta",  # Señorita
    "Dr",
    "Dra",
    "Prof",
    "Lic",  # Licenciado
    "Ing",  # Ingeniero
    "D",  # Don
    "Dña",  # Doña
    # Common abbreviations
    "etc",
    "pág",  # página
    "págs",  # páginas
    "núm",  # número
    "vol",
    "cap",  # capítulo
    "aprox",  # aproximadamente
    "máx",
    "mín",
    "Ud",  # Usted
    "Uds",  # Ustedes
    # Geographic
    "Av",  # Avenida
    "Pza",  # Plaza
    # Company types
    "SA",
    "Cía",  # Compañía
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Italian abbreviations
# =============================================================================
ITALIAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Sig",  # Signore
    "Dr",
    "Dott",  # Dottore
    "Prof",
    "Ing",
    "Avv",  # Avvocato
    "Rag",  # Ragioniere
    # Common abbreviations
    "ecc",  # eccetera
    "pag",  # pagina
    "pagg",  # pagine
    "vol",
    "cap",  # capitolo
    "num",  # numero
    "ca",  # circa
    "cfr",  # confronta
    # Geographic
    "Via",
    "Pza",  # Piazza
    # Company types
    "SpA",
    "Srl",
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Dutch abbreviations
# =============================================================================
DUTCH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Dhr",  # De heer
    "Mevr",  # Mevrouw
    "Dr",
    "Prof",
    "Mr",  # Meester (legal)
    "Ir",  # Ingenieur
    "Drs",  # Doctorandus
    # Common abbreviations
    "enz",  # enzovoort
    "etc",
    "bijv",  # bijvoorbeeld
    "ca",  # circa
    "max",
    "min",
    "nr",  # nummer
    "pag",  # pagina
    "vol",
    "z",  # z.g. (zogenaamd)
    "g",  # z.g.
    "o",  # o.a.
    "a",  # o.a., e.a.
    "e",  # e.a., e.d.
    "d",  # e.d.
    "m",  # m.b.t.
    "b",  # m.b.t.
    "t",  # m.b.t.
    # Geographic
    "Str",  # Straat
    # Company types
    "BV",
    "NV",
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Portuguese abbreviations
# =============================================================================
PORTUGUESE_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Sr",  # Senhor
    "Sra",  # Senhora
    "Dr",
    "Dra",
    "Prof",
    "Eng",  # Engenheiro
    # Common abbreviations
    "etc",
    "pág",  # página
    "págs",
    "núm",  # número
    "vol",
    "cap",  # capítulo
    "aprox",
    "máx",
    "mín",
    # Geographic
    "Av",  # Avenida
    "Pça",  # Praça
    # Company types
    "SA",
    "Lda",  # Limitada
    "Cia",  # Companhia
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Russian abbreviations
# =============================================================================
RUSSIAN_ABBREVIATIONS: set[str] = (
    {
        # Titles and honorifics
        "г",  # господин, год
        "гг",  # годы
        "гр",  # гражданин
        "д",  # доктор
        "др",  # другой, доктор
        "проф",  # профессор
        # Common abbreviations
        "т",  # так, том
        "е",  # т.е.
        "п",  # т.п.
        "н",  # н.э.
        "э",  # н.э.
        "см",  # смотри
        "ср",  # сравни
        "напр",  # например
        "и",  # и т.д.
        "в",  # в т.ч.
        "ч",  # в т.ч.
        "с",  # стр., с.
        "стр",  # страница
        "рис",  # рисунок
        "табл",  # таблица
        "гл",  # глава
        "ок",  # около
        "прим",  # примечание
        "млн",  # миллион
        "млрд",  # миллиард
        "тыс",  # тысяча
        # Geographic
        "ул",  # улица
        "пр",  # проспект
        "пл",  # площадь
        # дом
        "кв",  # квартира
        "обл",  # область
        "р",  # район
    }
    | _SINGLE_LETTERS_CYRILLIC
    | _SINGLE_LETTERS_LATIN
)

# =============================================================================
# Ukrainian abbreviations
# =============================================================================
UKRAINIAN_ABBREVIATIONS: set[str] = (
    {
        # Titles and honorifics
        "п",  # пан
        "д",  # доктор
        "проф",  # професор
        # Common abbreviations
        "т",  # так, том
        "і",  # і т.д.
        "н",  # н.е.
        "е",  # н.е.
        "див",  # дивись
        "напр",  # наприклад
        "с",  # стор., с.
        "стор",  # сторінка
        "рис",  # рисунок
        "табл",  # таблиця
        "розд",  # розділ
        "прим",  # примітка
        "млн",  # мільйон
        "млрд",  # мільярд
        "тис",  # тисяча
        # Geographic
        "вул",  # вулиця
        "пр",  # проспект
        "пл",  # площа
        "буд",  # будинок
        "кв",  # квартира
        "обл",  # область
        "р",  # район
    }
    | _SINGLE_LETTERS_CYRILLIC
    | _SINGLE_LETTERS_LATIN
)

# =============================================================================
# Greek abbreviations
# =============================================================================
GREEK_ABBREVIATIONS: set[str] = (
    {
        # Titles and honorifics
        "κ",  # κύριος
        "κα",  # κυρία
        "Δρ",  # Δόκτωρ
        # Common abbreviations
        "π",  # π.χ.
        "χ",  # π.χ.
        # κ.λπ.
        "λ",  # κ.λπ.
        "τ",  # κ.τ.λ.
        "σ",  # σελ.
        "σελ",  # σελίδα
        "βλ",  # βλέπε
        "αρ",  # αριθμός
        "εκ",  # εκατομμύριο
        # Geographic
        "οδ",  # οδός
        "λεωφ",  # λεωφόρος
        "πλ",  # πλατεία
    }
    | _SINGLE_LETTERS_GREEK
    | _SINGLE_LETTERS_LATIN
)

# =============================================================================
# Polish abbreviations
# =============================================================================
POLISH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "dr",  # doktor
    "prof",  # profesor
    "mgr",  # magister
    "inż",  # inżynier
    "ks",  # ksiądz
    # Common abbreviations
    "itd",  # i tak dalej
    "itp",  # i tym podobne
    "np",  # na przykład
    "tj",  # to jest
    "tzn",  # to znaczy
    "tzw",  # tak zwany
    "ok",  # około
    "s",  # strona
    "str",  # strona
    "r",  # rok
    "w",  # wiek
    "nr",  # numer
    "pkt",  # punkt
    "rozdz",  # rozdział
    "przyp",  # przypis
    "mln",  # milion
    "mld",  # miliard
    "tys",  # tysiąc
    # Geographic
    "ul",  # ulica
    "al",  # aleja
    "pl",  # plac
    "os",  # osiedle
    "woj",  # województwo
    "pow",  # powiat
    "gm",  # gmina
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Danish abbreviations
# =============================================================================
DANISH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "hr",  # herr
    "fru",  # fru
    "dr",  # doktor
    "prof",  # professor
    # Common abbreviations
    "osv",  # og så videre
    "mv",  # med videre
    "fx",  # for eksempel
    "dvs",  # det vil sige
    "mfl",  # med flere
    "ca",  # cirka
    "jf",  # jævnfør
    "pga",  # på grund af
    "s",  # side
    "nr",  # nummer
    "kap",  # kapitel
    "mio",  # million
    "mia",  # milliard
    # Geographic
    "vej",
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Swedish abbreviations
# =============================================================================
SWEDISH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "hr",  # herr
    "fru",  # fru
    "dr",  # doktor
    "prof",  # professor
    # Common abbreviations
    "osv",  # och så vidare
    "mm",  # med mera
    "tex",  # till exempel
    "t",  # t.ex.
    "ex",  # t.ex.
    "dvs",  # det vill säga
    "mfl",  # med flera
    "ca",  # cirka
    "jfr",  # jämför
    "pga",  # på grund av
    "s",  # sida
    "nr",  # nummer
    "kap",  # kapitel
    "milj",  # miljon
    "mdr",  # miljard
    # Geographic
    "gata",
    "v",  # väg
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Norwegian (Bokmal) abbreviations
# =============================================================================
NORWEGIAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "hr",  # herr
    "fru",  # fru
    "dr",  # doktor
    "prof",  # professor
    # Common abbreviations
    "osv",  # og så videre
    "mv",  # med videre
    "f",  # f.eks.
    "eks",  # f.eks.
    "dvs",  # det vil si
    "mfl",  # med flere
    "ca",  # cirka
    "jf",  # jfr
    "jfr",  # jamfør
    "pga",  # på grunn av
    "s",  # side
    "nr",  # nummer
    "kap",  # kapittel
    "mill",  # million
    "mrd",  # milliard
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Finnish abbreviations
# =============================================================================
FINNISH_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "hra",  # herra
    "rva",  # rouva
    "tri",  # tohtori
    "prof",  # professori
    # Common abbreviations
    "jne",  # ja niin edelleen
    "ym",  # ynnä muuta
    "esim",  # esimerkiksi
    "ts",  # toisin sanoen
    "mm",  # muun muassa
    "n",  # noin
    "vrt",  # vertaa
    "ks",  # katso
    "s",  # sivu
    "nro",  # numero
    "luku",
    "milj",  # miljoona
    "mrd",  # miljardi
    # Geographic
    "katu",
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Croatian abbreviations
# =============================================================================
CROATIAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "g",  # gospodin
    "gđa",  # gospođa
    "gđica",  # gospođica
    "dr",  # doktor
    "prof",  # profesor
    "mr",  # magistar
    # Common abbreviations
    "itd",  # i tako dalje
    "npr",  # na primjer
    "tj",  # to jest
    "tzv",  # takozvani
    "sl",  # slično
    "v",  # vidi
    "str",  # stranica
    "br",  # broj
    "god",  # godina
    "st",  # stoljeće
    "pogl",  # poglavlje
    # Geographic
    "ul",  # ulica
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Slovenian abbreviations
# =============================================================================
SLOVENIAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "g",  # gospod
    "ga",  # gospa
    "dr",  # doktor
    "prof",  # profesor
    "mag",  # magister
    # Common abbreviations
    "itd",  # in tako dalje
    "npr",  # na primer
    "tj",  # to je
    "t",  # t.i. (tako imenovan)
    "i",  # t.i.
    "ipd",  # in podobno
    "prim",  # primerjaj
    "gl",  # glej
    "str",  # stran
    "št",  # število
    "l",  # leto
    "pogl",  # poglavje
    "mio",  # milijon
    "mrd",  # milijarda
    # Geographic
    "ul",  # ulica
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Romanian abbreviations
# =============================================================================
ROMANIAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "dl",  # domnul
    "dna",  # doamna
    "dr",  # doctor
    "prof",  # profesor
    "ing",  # inginer
    # Common abbreviations
    "etc",  # et cetera
    "ex",  # exemplu
    "pag",  # pagina
    "nr",  # număr
    "vol",  # volum
    "cap",  # capitol
    "aprox",  # aproximativ
    # Geographic
    "str",  # strada
    "bd",  # bulevardul
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Lithuanian abbreviations
# =============================================================================
LITHUANIAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "p",  # ponas/ponia
    "dr",  # daktaras
    "prof",  # profesorius
    # Common abbreviations
    "t",  # t.y. (tai yra)
    "y",  # t.y.
    "pvz",  # pavyzdžiui
    "pan",  # panašiai
    "kt",  # kita
    "žr",  # žiūrėk
    "psl",  # puslapis
    "nr",  # numeris
    "sk",  # skyrius
    "mln",  # milijonas
    "mlrd",  # milijardas
    # Geographic
    "g",  # gatvė
    "pr",  # prospektas
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Macedonian abbreviations
# =============================================================================
MACEDONIAN_ABBREVIATIONS: set[str] = (
    {
        # Titles and honorifics
        "г",  # господин
        "гца",  # госпоѓа
        "д",  # доктор
        "проф",  # професор
        # Common abbreviations
        "т",  # т.е. (тоа е)
        "е",  # т.е.
        "итн",  # и така натаму
        "на",  # на пр.
        "пр",  # на пр.
        "в",  # види
        "стр",  # страница
        "бр",  # број
        "год",  # година
        "гл",  # глава
        # Geographic
        "ул",  # улица
    }
    | _SINGLE_LETTERS_CYRILLIC
    | _SINGLE_LETTERS_LATIN
)

# =============================================================================
# Catalan abbreviations
# =============================================================================
CATALAN_ABBREVIATIONS: set[str] = {
    # Titles and honorifics
    "Sr",  # Senyor
    "Sra",  # Senyora
    "Dr",
    "Dra",
    "Prof",
    # Common abbreviations
    "etc",
    "pàg",  # pàgina
    "núm",  # número
    "vol",
    "cap",  # capítol
    "aprox",  # aproximadament
    "màx",
    "mín",
    # Geographic
    "Av",  # Avinguda
    "Pl",  # Plaça
    # Company types
    "SA",
    "SL",
} | _SINGLE_LETTERS_LATIN

# =============================================================================
# Model to abbreviations mapping
# =============================================================================

# Mapping from spaCy model names to abbreviation sets
MODEL_TO_ABBREVIATIONS: dict[str, set[str]] = {
    # English models
    "en_core_web_sm": ENGLISH_ABBREVIATIONS,
    "en_core_web_md": ENGLISH_ABBREVIATIONS,
    "en_core_web_lg": ENGLISH_ABBREVIATIONS,
    "en_core_web_trf": ENGLISH_ABBREVIATIONS,
    # German models
    "de_core_news_sm": GERMAN_ABBREVIATIONS,
    "de_core_news_md": GERMAN_ABBREVIATIONS,
    "de_core_news_lg": GERMAN_ABBREVIATIONS,
    "de_dep_news_trf": GERMAN_ABBREVIATIONS,
    # French models
    "fr_core_news_sm": FRENCH_ABBREVIATIONS,
    "fr_core_news_md": FRENCH_ABBREVIATIONS,
    "fr_core_news_lg": FRENCH_ABBREVIATIONS,
    "fr_dep_news_trf": FRENCH_ABBREVIATIONS,
    # Spanish models
    "es_core_news_sm": SPANISH_ABBREVIATIONS,
    "es_core_news_md": SPANISH_ABBREVIATIONS,
    "es_core_news_lg": SPANISH_ABBREVIATIONS,
    "es_dep_news_trf": SPANISH_ABBREVIATIONS,
    # Italian models
    "it_core_news_sm": ITALIAN_ABBREVIATIONS,
    "it_core_news_md": ITALIAN_ABBREVIATIONS,
    "it_core_news_lg": ITALIAN_ABBREVIATIONS,
    # Dutch models
    "nl_core_news_sm": DUTCH_ABBREVIATIONS,
    "nl_core_news_md": DUTCH_ABBREVIATIONS,
    "nl_core_news_lg": DUTCH_ABBREVIATIONS,
    # Portuguese models
    "pt_core_news_sm": PORTUGUESE_ABBREVIATIONS,
    "pt_core_news_md": PORTUGUESE_ABBREVIATIONS,
    "pt_core_news_lg": PORTUGUESE_ABBREVIATIONS,
    # Russian models
    "ru_core_news_sm": RUSSIAN_ABBREVIATIONS,
    "ru_core_news_md": RUSSIAN_ABBREVIATIONS,
    "ru_core_news_lg": RUSSIAN_ABBREVIATIONS,
    # Ukrainian models
    "uk_core_news_sm": UKRAINIAN_ABBREVIATIONS,
    "uk_core_news_md": UKRAINIAN_ABBREVIATIONS,
    "uk_core_news_lg": UKRAINIAN_ABBREVIATIONS,
    "uk_core_news_trf": UKRAINIAN_ABBREVIATIONS,
    # Greek models
    "el_core_news_sm": GREEK_ABBREVIATIONS,
    "el_core_news_md": GREEK_ABBREVIATIONS,
    "el_core_news_lg": GREEK_ABBREVIATIONS,
    # Polish models
    "pl_core_news_sm": POLISH_ABBREVIATIONS,
    "pl_core_news_md": POLISH_ABBREVIATIONS,
    "pl_core_news_lg": POLISH_ABBREVIATIONS,
    # Danish models
    "da_core_news_sm": DANISH_ABBREVIATIONS,
    "da_core_news_md": DANISH_ABBREVIATIONS,
    "da_core_news_lg": DANISH_ABBREVIATIONS,
    "da_core_news_trf": DANISH_ABBREVIATIONS,
    # Swedish models
    "sv_core_news_sm": SWEDISH_ABBREVIATIONS,
    "sv_core_news_md": SWEDISH_ABBREVIATIONS,
    "sv_core_news_lg": SWEDISH_ABBREVIATIONS,
    # Norwegian Bokmal models
    "nb_core_news_sm": NORWEGIAN_ABBREVIATIONS,
    "nb_core_news_md": NORWEGIAN_ABBREVIATIONS,
    "nb_core_news_lg": NORWEGIAN_ABBREVIATIONS,
    # Finnish models
    "fi_core_news_sm": FINNISH_ABBREVIATIONS,
    "fi_core_news_md": FINNISH_ABBREVIATIONS,
    "fi_core_news_lg": FINNISH_ABBREVIATIONS,
    # Croatian models
    "hr_core_news_sm": CROATIAN_ABBREVIATIONS,
    "hr_core_news_md": CROATIAN_ABBREVIATIONS,
    "hr_core_news_lg": CROATIAN_ABBREVIATIONS,
    # Slovenian models
    "sl_core_news_sm": SLOVENIAN_ABBREVIATIONS,
    "sl_core_news_md": SLOVENIAN_ABBREVIATIONS,
    "sl_core_news_lg": SLOVENIAN_ABBREVIATIONS,
    "sl_core_news_trf": SLOVENIAN_ABBREVIATIONS,
    # Romanian models
    "ro_core_news_sm": ROMANIAN_ABBREVIATIONS,
    "ro_core_news_md": ROMANIAN_ABBREVIATIONS,
    "ro_core_news_lg": ROMANIAN_ABBREVIATIONS,
    # Lithuanian models
    "lt_core_news_sm": LITHUANIAN_ABBREVIATIONS,
    "lt_core_news_md": LITHUANIAN_ABBREVIATIONS,
    "lt_core_news_lg": LITHUANIAN_ABBREVIATIONS,
    # Macedonian models
    "mk_core_news_sm": MACEDONIAN_ABBREVIATIONS,
    "mk_core_news_md": MACEDONIAN_ABBREVIATIONS,
    "mk_core_news_lg": MACEDONIAN_ABBREVIATIONS,
    # Catalan models
    "ca_core_news_sm": CATALAN_ABBREVIATIONS,
    "ca_core_news_md": CATALAN_ABBREVIATIONS,
    "ca_core_news_lg": CATALAN_ABBREVIATIONS,
    "ca_core_news_trf": CATALAN_ABBREVIATIONS,
    # Note: Chinese (zh), Japanese (ja), Korean (ko) don't use period-based
    # abbreviations in the same way, so they have no entries here.
    # The function will return an empty set for these languages.
}

# Language-level fallback for model families and future package versions.  The
# exact model map above remains authoritative when a package is known.
LANGUAGE_TO_ABBREVIATIONS: dict[str, set[str]] = {
    "ca": CATALAN_ABBREVIATIONS,
    "da": DANISH_ABBREVIATIONS,
    "de": GERMAN_ABBREVIATIONS,
    "el": GREEK_ABBREVIATIONS,
    "en": ENGLISH_ABBREVIATIONS,
    "es": SPANISH_ABBREVIATIONS,
    "fi": FINNISH_ABBREVIATIONS,
    "fr": FRENCH_ABBREVIATIONS,
    "hr": CROATIAN_ABBREVIATIONS,
    "it": ITALIAN_ABBREVIATIONS,
    "lt": LITHUANIAN_ABBREVIATIONS,
    "mk": MACEDONIAN_ABBREVIATIONS,
    "nb": NORWEGIAN_ABBREVIATIONS,
    "nl": DUTCH_ABBREVIATIONS,
    "pl": POLISH_ABBREVIATIONS,
    "pt": PORTUGUESE_ABBREVIATIONS,
    "ro": ROMANIAN_ABBREVIATIONS,
    "ru": RUSSIAN_ABBREVIATIONS,
    "sl": SLOVENIAN_ABBREVIATIONS,
    "sv": SWEDISH_ABBREVIATIONS,
    "uk": UKRAINIAN_ABBREVIATIONS,
}

# =============================================================================
# Sentence-ending abbreviations
# =============================================================================

# Abbreviations that commonly appear at the END of sentences
# These should NOT trigger a merge with the following sentence
# even though they are abbreviations
# Note: Jr/Sr are excluded because in some languages (Spanish, Portuguese)
# they're titles that precede names, not suffixes that follow names
SENTENCE_ENDING_ABBREVIATIONS: set[str] = {
    # English company suffixes
    "etc",  # "...and so on, etc. The next topic..."
    "Inc",  # "...by Apple, Inc. The company..."
    "Corp",  # "...by Microsoft Corp. They announced..."
    "Ltd",  # "...by Acme Ltd. The firm..."
    "Co",  # "...by Acme Co. They manufacture..."
    # Units that often end sentences
    "in",  # "...measured 5 in. The result..."
    "ft",  # "...about 10 ft. The building..."
    "oz",  # "...weighs 8 oz. The package..."
    "lb",  # "...weighs 50 lb. The shipment..."
    "yr",  # "...lasted 5 yr. The study..."
    "mo",  # "...took 6 mo. The project..."
}

# =============================================================================
# Sentence starters (multi-language)
# =============================================================================

# Common sentence starters (mostly language-independent)
# These are words that typically start new sentences
SENTENCE_STARTERS: set[str] = {
    # English
    "The",
    "A",
    "An",
    "This",
    "That",
    "These",
    "Those",
    "It",
    "There",
    "Here",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "How",
    "If",
    "Although",
    "Because",
    "Since",
    "While",
    "After",
    "Before",
    "However",
    "Therefore",
    "Moreover",
    "Furthermore",
    "In",
    "On",
    "At",
    "For",
    "But",
    "And",
    "Or",
    "So",
    "Yet",
    "As",
    "We",
    "They",
    "He",
    "She",
    "You",
    "I",
    # German
    "Der",
    "Die",
    "Das",
    "Ein",
    "Eine",
    "Es",
    "Er",
    "Sie",
    "Wir",
    "Ich",
    "Wenn",
    "Weil",
    "Obwohl",
    "Aber",
    "Und",
    "Oder",
    "Denn",
    "Dann",
    "Hier",
    "Dort",
    "Was",
    "Wer",
    "Wie",
    "Wo",
    "Warum",
    "Jedoch",
    "Daher",
    "Deshalb",
    "Außerdem",
    # French
    "Le",
    "La",
    "Les",
    "Un",
    "Une",
    "Il",
    "Elle",
    "Ils",
    "Elles",
    "Nous",
    "Je",
    "Ce",
    "Cet",
    "Cette",
    "Ces",
    "Quand",
    "Parce",
    "Mais",
    "Et",
    "Ou",
    "Donc",
    "Ici",
    "Qui",
    "Que",
    "Quoi",
    "Comment",
    "Pourquoi",
    "Cependant",
    # Spanish
    "El",
    "Los",
    "Él",
    "Ella",
    "Ellos",
    "Ellas",
    "Nosotros",
    "Yo",
    "Este",
    "Esta",
    "Estos",
    "Estas",
    "Ese",
    "Esa",
    "Cuando",
    "Porque",
    "Pero",
    "Y",
    "Aquí",
    "Allí",
    "Qué",
    "Quién",
    "Cómo",
    "Dónde",
    "Por",
    "Sin",
    # Italian
    "Lo",
    "Gli",
    "Lui",
    "Lei",
    "Loro",
    "Noi",
    "Io",
    "Questo",
    "Questa",
    "Questi",
    "Queste",
    "Quello",
    "Quella",
    "Quando",
    "Perché",
    "Ma",
    "Chi",
    "Che",
    "Come",
    "Dove",
    "Tuttavia",
    # Dutch
    "De",
    "Het",
    "Een",
    "Hij",
    "Zij",
    "Wij",
    "Ik",
    "Dit",
    "Dat",
    "Deze",
    "Wanneer",
    "Omdat",
    "Maar",
    "Waar",
    "Wat",
    "Hoe",
    "Waarom",
    "Echter",
    # Portuguese
    "O",
    "Os",
    "Ele",
    "Ela",
    "Eles",
    "Elas",
    "Nós",
    "Eu",
    "Isto",
    "Isso",
    "Aquilo",
    "Mas",
    "Aqui",
    "Ali",
    "Quem",
    "Como",
    "Onde",
    "Porém",
    # Russian
    "Это",
    "Он",
    "Она",
    "Они",
    "Мы",
    "Я",
    "Что",
    "Когда",
    "Где",
    "Как",
    "Почему",
    "Если",
    "Хотя",
    "Потому",
    "Но",
    "И",
    "Или",
    "Однако",
    "Поэтому",
    "Здесь",
    "Там",
    # Polish
    "To",
    "Ten",
    "Ta",
    "Ci",
    "Te",
    "Ona",
    "Oni",
    "My",
    "Ja",
    "Co",
    "Kiedy",
    "Gdzie",
    "Jak",
    "Dlaczego",
    "Jeśli",
    "Chociaż",
    "Ponieważ",
    "Ale",
    "Jednak",
    "Dlatego",
    "Tutaj",
    "Tam",
}


def get_abbreviations(
    language_model: str | None = None,
    *,
    language: str | None = None,
) -> set[str]:
    """Get abbreviations using exact model, package language, or language.

    Args:
        language_model: Optional exact spaCy model name. ``None``, ``""`` and
            ``"auto"`` use language-based lookup.
        language: Optional independent language hint, such as ``"de"``.

    Returns:
        Set of abbreviations for that language, or empty set if not supported
    """
    if language_model:
        normalized_model = language_model.strip().lower().replace("-", "_")
        if normalized_model in MODEL_TO_ABBREVIATIONS:
            return MODEL_TO_ABBREVIATIONS[normalized_model]
        if normalized_model not in {"auto", "none"}:
            match = re.match(r"^(?P<language>[a-z]{2,3})_", normalized_model)
            if match:
                inferred = normalize_spacy_language(match.group("language"))
                return LANGUAGE_TO_ABBREVIATIONS.get(inferred, set())

    if language:
        normalized_language = normalize_spacy_language(language)
        return LANGUAGE_TO_ABBREVIATIONS.get(normalized_language, set())

    return set()


def get_sentence_starters() -> set[str]:
    """Get the set of common sentence starters.

    Returns:
        Set of words that typically start sentences
    """
    return SENTENCE_STARTERS


def get_sentence_ending_abbreviations() -> set[str]:
    """Get the set of abbreviations that commonly end sentences.

    These are abbreviations like "etc.", "Inc.", "Ltd." that frequently
    appear at the end of sentences and should NOT trigger a merge with
    the following sentence.

    Returns:
        Set of abbreviations that can end sentences
    """
    return SENTENCE_ENDING_ABBREVIATIONS
