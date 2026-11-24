import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

from dnp_rag import config, indice, rag
from dnp_rag import recuperacion as rec
from dnp_rag.evaluacion import similitudes_maximas


@pytest.fixture()
def vs(corpus, embeddings):
    return indice.construir_indice(corpus, embeddings)


# ------------------------------------------------------------------ detectar el gobierno de la pregunta
@pytest.mark.parametrize("pregunta,esperado", [
    ("plan de Santos 2014-2018", ["2014-2018"]),            # un rango explícito manda sobre el nombre
    ("plan de Santos", ["2010-2014", "2014-2018"]),
    ("PND 2018-2022", ["2018-2022"]),                       # no debe sumar 2022-2026
    ("qué se hizo en 2016", ["2014-2018"]),
    ("Compara Uribe y Duque", ["2006-2010", "2018-2022"]),
    ("sin periodo", []),
])
def test_detectar_periodos(pregunta, esperado):
    assert rec.detectar_periodos(pregunta) == esperado


# ------------------------------------------------------------------ recuperación densa (FAISS)
def test_la_busqueda_densa_encuentra_el_tema_y_la_similitud_es_coseno(vs, corpus):
    evs = rec.buscar_denso(vs, "subsidios de vivienda de interés social y crédito hipotecario", k=5)
    assert len(evs) == 5 and all(e["tema"] == "Vivienda" for e in evs[:3])
    assert evs[0]["similitud"] >= evs[-1]["similitud"] and all(0 <= e["similitud"] <= 1.0001 for e in evs)
    exacto = rec.buscar_denso(vs, evs[0]["texto"], k=1)[0]
    assert exacto["similitud"] == pytest.approx(1.0, abs=1e-3)          # un texto con sí mismo: coseno 1


def test_filtrar_por_gobierno_devuelve_k_aunque_el_corpus_este_sesgado(embeddings):
    """Regresión: LangChain filtra DESPUÉS de traer `fetch_k` vecinos (20 por defecto). Si casi todo lo cercano
    es de otro gobierno, el filtro devolvía menos de k resultados, o ninguno."""
    filas = [{"id": i, "periodo": "2006-2010", "presidente": "X", "documento": "a", "pagina": i,
              "texto": f"vivienda de interés social subsidios habitacionales caso {i}"} for i in range(300)]
    filas += [{"id": 1000 + i, "periodo": "2022-2026", "presidente": "Y", "documento": "b", "pagina": i,
               "texto": f"política exterior comercio internacional cooperación caso {i} y vivienda"} for i in range(8)]
    vs = indice.construir_indice(pd.DataFrame(filas), embeddings)
    evs = rec.buscar_denso(vs, "vivienda de interés social subsidios habitacionales", k=5, periodos=["2022-2026"])
    assert len(evs) == 5 and {e["periodo"] for e in evs} == {"2022-2026"}


def test_el_indice_se_reutiliza_y_se_reconstruye_si_cambia_el_corpus(corpus, embeddings, datos_tmp, monkeypatch):
    vs1 = indice.cargar_o_construir(corpus, embeddings)
    assert (config.INDICE_DIR / "faiss_pnd" / "meta.json").exists()
    with monkeypatch.context() as m:                                 # la 2.ª vez no debe reconstruir
        m.setattr(indice, "construir_indice", lambda *a, **k: pytest.fail("debía usar el índice guardado"))
        vs2 = indice.cargar_o_construir(corpus, embeddings)
    q = "desempleo e informalidad laboral"
    assert [e["id"] if "id" in e else e["texto"] for e in rec.buscar_denso(vs1, q, 3)] == \
           [e["id"] if "id" in e else e["texto"] for e in rec.buscar_denso(vs2, q, 3)]
    assert indice.cargar_o_construir(corpus.iloc[:60], embeddings).index.ntotal == 60   # corpus distinto -> reconstruye


# ------------------------------------------------------------------ TF-IDF (línea base de la S13_2)
def test_tfidf_filtra_antes_de_ordenar_y_no_inventa(corpus):
    modelo = rec.crear_tfidf(corpus)
    evs = rec.buscar_tfidf(modelo, "subsidios de vivienda y crédito hipotecario", k=8, periodos=["2022-2026"])
    assert len(evs) == 8 and {e["periodo"] for e in evs} == {"2022-2026"} and evs[0]["tema"] == "Vivienda"
    assert rec.buscar_tfidf(modelo, "zzzz qqqq", k=3) == []


# ------------------------------------------------------------------ RAG
def test_respuesta_extractiva_con_citas_y_fuentes(vs, embeddings):
    r = rag.responder("¿Qué se propone sobre vivienda de interés social y subsidios habitacionales?", vs, embeddings, umbral=0.3)
    assert r["suficiente"] and "[" in r["respuesta"] and r["evidencias"][0]["tema"] == "Vivienda"
    texto = rag.texto_chat(r)
    assert "Fuentes" in texto and "PND" in texto and "p." in texto


def test_el_umbral_rechaza_las_preguntas_fuera_de_dominio(vs, embeddings):
    sim_ok = similitudes_maximas(lambda q, k, p: rec.buscar_denso(vs, q, k, p), ["subsidios de vivienda de interés social"])
    sim_mal = similitudes_maximas(lambda q, k, p: rec.buscar_denso(vs, q, k, p), ["receta del sancocho con plátano y yuca"])
    assert sim_ok[0] > 0.3 > sim_mal[0]
    fuera = rag.responder("¿Cuál es la receta del sancocho con plátano y yuca?", vs, embeddings, umbral=0.3)
    assert not fuera["suficiente"] and fuera["respuesta"] == rag.SIN_EVIDENCIA


def test_pregunta_muy_corta_pide_detalle(vs, embeddings):
    r = rag.responder("vivienda", vs, embeddings)
    assert r["respuesta"] == rag.PEDIR_DETALLE and not r["suficiente"]


def test_si_la_pregunta_nombra_un_gobierno_solo_se_busca_en_ese_gobierno(vs, embeddings):
    r = rag.responder("¿Qué dice el plan 2018-2022 sobre salud, hospitales y vacunación?", vs, embeddings, umbral=0.1)
    assert r["suficiente"] and {e["periodo"] for e in r["evidencias"]} == {"2018-2022"}


def test_el_umbral_calibrado_se_guarda_y_se_carga(datos_tmp):
    assert rag.cargar_umbral() == config.UMBRAL_SIMILITUD
    rag.guardar_umbral(0.83)
    assert rag.cargar_umbral() == pytest.approx(0.83)


def test_cadena_langchain_con_llm_falso(vs, embeddings):
    """La cadena de la clase S14_1 (retriever | formatear -> prompt -> llm) con un LLM de mentira."""
    bien = FakeListLLM(responses=["Se plantean subsidios [1] y crédito [2]."])
    r = rag.responder("subsidios de vivienda de interés social", vs, embeddings, llm=bien, umbral=0.3)
    assert r["respuesta"].startswith("Se plantean") and "Aviso" not in r["respuesta"]
    mal = FakeListLLM(responses=["Respuesta sin ninguna cita."])
    assert "Aviso" in rag.responder("subsidios de vivienda de interés social", vs, embeddings, llm=mal, umbral=0.3)["respuesta"]
    assert rag.citas_validas("ver [1] y [3]", 5) and not rag.citas_validas("ver [9]", 5) and not rag.citas_validas("sin citas", 5)


def test_la_cadena_con_filtro_solo_usa_el_gobierno_pedido(vs, embeddings):
    visto = {}

    class Espia(FakeListLLM):
        def _call(self, prompt, *a, **k):
            visto["prompt"] = prompt
            return "ok [1]"

    rag.responder("¿Qué propone el plan 2010-2014 sobre desempleo, informalidad y mercado laboral?", vs, embeddings, llm=Espia(responses=["x"]), umbral=0.1)
    assert "(PND 2010-2014" in visto["prompt"] and "(PND 2022-2026" not in visto["prompt"]
    assert "CONTEXTO:" in visto["prompt"] and "[1] (PND" in visto["prompt"]
