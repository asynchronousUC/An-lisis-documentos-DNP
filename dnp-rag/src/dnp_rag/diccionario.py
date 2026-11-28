"""Diccionario de palabras clave por tema (viene de la v4).

Las etiquetas las produce una REGLA, no una persona. Por eso NO se usan
para entrenar ni evaluar ningún modelo, ni se reporta ninguna "exactitud" contra ellas (esa fue la
crítica a la v4: un modelo que predice estas etiquetas solo aprende la regla). Aquí sirven únicamente
para explorar el corpus y como filtro opcional de búsqueda. La calidad del sistema se mide con
`evaluacion.py` contra las preguntas anotadas por personas (data/gold).
"""
import re
import unicodedata

OTROS = "Otros / sin tema claro"

# Patrones sobre texto SIN tildes y en minúsculas (ñ -> n). \w* = cualquier terminación.
TEMAS = {
 "Educación": [r"\beducaci\w+", r"\beducativ\w+", r"\bescolar\w*", r"\bescuel\w+", r"\bcolegi\w+",
    r"\bdocentes?\b", r"\bmaestr[oa]s?\b", r"\bestudiantes?\b", r"\buniversidad\w*", r"\buniversitari\w+",
    r"\bsena\b", r"\bicetex\b", r"\bpedagog\w+", r"\bensenanza\b", r"\baprendizaje\w*", r"\bcurricul\w+",
    r"\balfabetiz\w+", r"\banalfabet\w+", r"\bdesercion\b", r"\bprimaria\b", r"\bsecundaria\b",
    r"\bbachiller\w*", r"\bjornada unica\b", r"\bformacion (?:tecnica|profesional|para el trabajo)\b"],
 "Vivienda": [r"\bvivienda\w*", r"\bhabitacional\w*", r"\bhabitat\b", r"\bvis\b", r"\bvip\b",
    r"\bmi casa ya\b", r"\bhipotecari\w+", r"\bupac\b", r"\bleasing habitacional\b",
    r"\bmejoramiento de barrios\b", r"\bdeficit (?:cuantitativo|cualitativo|habitacional)\b",
    r"\basentamientos?\b", r"\bbarrios? (?:subnormales|informales)\b"],
 "Empleo": [r"\bempleos?\b(?!\s+de\s+(?:los|las|la|el)?\s*(?:recursos|fondos|medios|fuerza))",
    r"\bdesempleo\b", r"\bsubempleo\b", r"\bempleabilidad\b", r"\bocupad[oa]s?\b", r"\btasa de ocupacion\b",
    r"\bmercado laboral\b", r"\blaboral\w*", r"\btrabajadores?\b", r"\btrabajo decente\b",
    r"\binformalidad\b", r"\bsalari\w+", r"\bsindical\w*", r"\bpoblacion economicamente activa\b"],
 "Salud": [r"\bsalud\b", r"\bhospital\w*", r"\bsanitari\w+", r"\bmedic\w+", r"\benfermedad\w*",
    r"\bmortalidad\b", r"\bvacun\w+", r"\bepidemi\w+", r"\bpandemi\w+", r"\bnutricion\w*",
    r"\bdesnutricion\b", r"\bregimen (?:subsidiado|contributivo)\b", r"\beps\b", r"\bips\b", r"\bpacientes?\b"],
 "Seguridad y justicia": [r"\bseguridad (?:ciudadana|humana|nacional|publica)\b", r"\bcriminal\w*",
    r"\bdelit\w+", r"\bhomicidio\w*", r"\bpolicia\w*", r"\bfuerza publica\b", r"\bfuerzas militares\b",
    r"\bnarcotrafico\b", r"\bjusticia\b", r"\bjudicial\w*", r"\bcarcel\w*", r"\bpenitenciari\w+",
    r"\bcorrupcion\b", r"\bterroris\w+", r"\bcrimen\w*", r"\bimpunidad\b", r"\bsecuestr\w+", r"\bconvivencia\b"],
 "Paz, conflicto y víctimas": [r"\bpaz\b", r"\bvictimas?\b", r"\bconflicto armado\b", r"\bposconflicto\b",
    r"\bdesplazad\w+", r"\breparacion\w*", r"\brestitucion\b", r"\bdesmovilizacion\b", r"\breintegracion\b",
    r"\breincorporacion\b", r"\bgrupos armados\b", r"\bfarc\b", r"\bparamilitar\w*", r"\bacuerdo de paz\b"],
 "Infraestructura y transporte": [r"\binfraestructura\w*", r"\bvias?\b", r"\bcarretera\w*", r"\btransporte\w*",
    r"\bmovilidad\b", r"\baeropuerto\w*", r"\bpuertos?\b", r"\bferrovia\w*", r"\bferrocarril\w*",
    r"\bconcesion\w*", r"\b4g\b", r"\btroncal\w*", r"\bmetro\b", r"\bvial\w*", r"\blogistic\w+"],
 "Medio ambiente y cambio climático": [r"\bambient\w+", r"\bclimatic\w+", r"\bcambio climatico\b",
    r"\bbiodiversidad\b", r"\bdeforestacion\b", r"\bbosques?\b", r"\becosistem\w+", r"\bemisiones\b",
    r"\bgases de efecto invernadero\b", r"\bcarbono\b", r"\bresiduos\b", r"\bareas protegidas\b",
    r"\brecursos naturales\b", r"\bcuencas?\b", r"\bcontaminacion\b", r"\breciclaj\w+",
    r"\bgestion del riesgo\b", r"\bdesastres?\b"],
 "Agua potable y saneamiento": [r"\bacueducto\w*", r"\balcantarillado\b", r"\bagua potable\b",
    r"\bsaneamiento\b", r"\bresiduos solidos\b", r"\bplantas? de tratamiento\b"],
 "Agro y desarrollo rural": [r"\bagropecuari\w+", r"\bagricol\w+", r"\bagricultur\w+", r"\brural\w*",
    r"\bcampesin\w+", r"\bcultivos?\b", r"\bganader\w+", r"\bcatastro\b", r"\bbaldios\b",
    r"\bseguridad alimentaria\b", r"\bsoberania alimentaria\b", r"\bderecho humano a la alimentacion\b",
    r"\bpesca\w*", r"\bcafe\w*", r"\briego\b", r"\bagroindustri\w+", r"\breforma agraria\b", r"\bagrari\w+"],
 "Pobreza y protección social": [r"\bpobreza\b", r"\bpobres?\b", r"\bvulnerab\w+", r"\bproteccion social\b",
    r"\bsubsidios?\b", r"\btransferencias? monetarias?\b", r"\bfamilias en accion\b", r"\bsisben\b",
    r"\bred unidos\b", r"\bpension\w*", r"\binclusion social\b", r"\bprimera infancia\b", r"\bicbf\b",
    r"\bninez\b", r"\badulto mayor\b", r"\bdiscapacidad\b"],
 "Equidad, género y grupos étnicos": [r"\bgenero\b", r"\bmujer(?:es)?\b", r"\bequidad\b", r"\bdesigualdad\w*",
    r"\betnic\w+", r"\bindigena\w*", r"\bafrocolombian\w+", r"\bdiscriminacion\b", r"\blgbt\w*", r"\bdiversidad\b"],
 "Ciencia, tecnología e innovación": [r"\bciencia\b", r"\btecnologi\w+", r"\binnovaci\w+", r"\binvestigaci\w+",
    r"\bcolciencias\b", r"\bminciencias\b", r"\bdigital\w*", r"\bconectividad\b", r"\binternet\b",
    r"\btics?\b", r"\bbanda ancha\b", r"\btelecomunicaci\w+", r"\bgobierno en linea\b", r"\binteligencia artificial\b"],
 "Energía y minería": [r"\benergi\w+", r"\belectric\w+", r"\bminer\w+", r"\bmineroenergetic\w+",
    r"\bhidrocarburo\w*", r"\bpetroleo\b", r"\bgas natural\b", r"\bcarbon\b", r"\brenovable\w*",
    r"\bsolar\b", r"\beolic\w+", r"\bregalias\b", r"\becopetrol\b", r"\bminas?\b", r"\bsubsuelo\b"],
 "Competitividad, industria, comercio y turismo": [r"\bcompetitiv\w+", r"\bindustri\w+", r"\bexportaci\w+",
    r"\bimportaci\w+", r"\bcomercio\b", r"\bempresa\w*", r"\bempresarial\w*", r"\bemprendimiento\w*",
    r"\bpymes?\b", r"\bmipymes?\b", r"\binversion extranjera\b", r"\btlc\b", r"\baranceles?\b",
    r"\bzonas francas?\b", r"\bproductividad\b", r"\binternacionalizaci\w+", r"\bturismo\b", r"\bturistic\w+"],
 "Macroeconomía y finanzas públicas": [r"\bfiscal\w*", r"\bpresupuest\w+", r"\bdeuda\b", r"\bpib\b",
    r"\binflaci\w+", r"\btributari\w+", r"\bimpuestos?\b", r"\bgasto publico\b", r"\bbalanza de pagos\b",
    r"\btasa de cambio\b", r"\bregla fiscal\b", r"\bahorro\b", r"\binversion publica\b",
    r"\bsistema general de participaciones\b", r"\bsgp\b", r"\bcrecimiento economico\b",
    r"\bmacroeconom\w+", r"\bbanco de la republica\b"],
 "Gobierno, descentralización y participación": [r"\bdescentraliz\w+", r"\bentidades territoriales\b",
    r"\bmunicipios?\b", r"\bordenamiento territorial\b", r"\bparticipacion ciudadana\b", r"\btransparencia\b",
    r"\brendicion de cuentas\b", r"\bmodernizacion del estado\b", r"\badministracion publica\b",
    r"\bgestion publica\b", r"\bbuen gobierno\b", r"\bcontrol social\b", r"\bservicio civil\b"],
 "Cultura, deporte y recreación": [r"\bcultur\w+", r"\bpatrimonio\b", r"\bdeporte\w*", r"\bdeportiv\w+",
    r"\brecreaci\w+", r"\bbibliotec\w+", r"\bmuseos?\b", r"\bartistic\w+", r"\bartes\b",
    r"\bcreativ\w+", r"\beconomia naranja\b", r"\blectura\b"],
}
# Compila un regex por tema uniendo todos sus patrones con OR


TEMAS_RX = {t: re.compile("|".join(f"(?:{p})" for p in ps)) for t, ps in TEMAS.items()}
MIN_COINCIDENCIAS = 2   # con menos coincidencias de palabras clave, el párrafo queda como "Otros"


def sin_tildes(texto):
    return "".join(c for c in unicodedata.normalize("NFKD", texto.lower()) if not unicodedata.combining(c))


def clasificar_tema(texto):
    """Devuelve (tema, n.º de coincidencias). El tema con más coincidencias gana; el empate lo gana el primero."""
    t = sin_tildes(texto)
    conteo = {tema: len(rx.findall(t)) for tema, rx in TEMAS_RX.items()}
    mejor = max(conteo, key=conteo.get)            # max devuelve el primero en caso de empate
    if conteo[mejor] < MIN_COINCIDENCIAS:
        return OTROS, 0
    return mejor, conteo[mejor]


def etiquetar(df):
    """Agrega las columnas `tema` y `tema_puntaje` (etiquetas de EXPLORACIÓN, no verdad de referencia)."""
    res = [clasificar_tema(t) for t in df["texto"]]
    out = df.copy()
    out["tema"] = [r[0] for r in res]
    out["tema_puntaje"] = [r[1] for r in res]
    return out


def resumen_temas(df):
    r = df["tema"].value_counts().rename_axis("tema").reset_index(name="parrafos")
    r["porcentaje"] = (100 * r["parrafos"] / len(df)).round(1)
    return r
