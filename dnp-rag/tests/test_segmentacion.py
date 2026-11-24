import re

from dnp_rag import config, diccionario
from dnp_rag.segmentacion import (dividir_por_oraciones, es_indice, es_tabla, segmentar_documento,
                                  separar_oraciones)


def oracion(n_caracteres, palabra):
    return ((palabra + " ") * (n_caracteres // (len(palabra) + 1))).strip() + "."


def norm(s):
    return re.sub(r"\s+", " ", s)


def test_parrafo_largo_se_divide_sin_perder_texto():
    texto = " ".join(f"Esta es la oración número {i} del párrafo de prueba, con palabras para contar." for i in range(60))
    trozos = dividir_por_oraciones(texto)
    assert len(trozos) > 1 and norm(" ".join(trozos)) == norm(texto)


def test_trozo_final_corto_con_dos_trozos():
    """Con la línea vieja `trozos[-2] += ... + trozos.pop()` esto daba IndexError."""
    texto = oracion(1990, "Alfa") + " Fin corta del texto."
    assert norm(" ".join(dividir_por_oraciones(texto))) == norm(texto)


def test_trozo_final_corto_con_varios_trozos_no_pierde_texto():
    """Con 3 o más trozos la línea vieja no fallaba, pero sobrescribía un trozo y se perdía texto en silencio."""
    texto = " ".join([oracion(1900, "Alfa"), oracion(1900, "Beta"), oracion(1990, "Gamma"), "Fin corta del texto."])
    trozos = dividir_por_oraciones(texto)
    assert len(trozos) >= 3 and norm(" ".join(trozos)) == norm(texto)


def test_abreviaturas_no_cortan_oraciones():
    assert len(separar_oraciones("Según el Art. 5 del plan, EE. UU. apoyará. Luego J. M. Santos firmó el decreto. Fin.")) == 3


def test_tabla_e_indice():
    tabla = "2010 2011 2012 12,5 13,1 14,2 15,0 16,3 17,8 18,9 19,1 20,2 21,3 22,4 23,5 24,6 25,7 26,8"
    prosa = "El gobierno ampliará la cobertura de los servicios públicos en las regiones más apartadas del país. " * 3
    assert es_tabla(tabla) and not es_tabla(prosa)
    assert es_indice("Contenido\nIntroducción ........ 5\nCapítulo 1 ........ 9\nCapítulo 2 ........ 20")


def test_segmentar_documento_descarta_ruido():
    prosa = " ".join(["El gobierno fortalecerá la educación escolar y la formación de docentes en las regiones del país."] * 3)
    paginas = [(1, prosa), (2, "12"), (3, "Contenido\nCap 1 ..... 5\nCap 2 ..... 9\nCap 3 ..... 12"), (4, "texto corto")]
    segmentos, excluidas = segmentar_documento(paginas)
    assert [s["pagina"] for s in segmentos] == [1]
    assert [e["motivo"] for e in excluidas] == ["indice"]


def test_diccionario_marca_tema_y_otros():
    tema, puntaje = diccionario.clasificar_tema("El déficit habitacional y la vivienda de interés social requieren crédito hipotecario.")
    assert tema == "Vivienda" and puntaje >= 2
    assert diccionario.clasificar_tema("Texto sobre política exterior sin palabras clave relevantes.")[0] == diccionario.OTROS
    assert len(diccionario.TEMAS) == 18 and config.MIN_PALABRAS == 20
