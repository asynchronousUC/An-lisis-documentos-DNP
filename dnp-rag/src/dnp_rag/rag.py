"""Etapa 6: el RAG. Pregunta -> recuperar párrafos -> ¿hay evidencia suficiente? -> respuesta con citas.

Piezas de clase:
  - S13_2: respuesta EXTRACTIVA con citas (frases literales de los PND; no puede inventar).
  - S14_1: cadena de LangChain  {"context": retriever | formatear, "question": RunnablePassthrough()} | prompt | llm
           con un LLM de Hugging Face (TinyLlama) envuelto en HuggingFacePipeline.

Dos decisiones propias:
  - Umbral de similitud: si el mejor párrafo no se parece lo suficiente a la pregunta, el sistema dice
    "no hay evidencia" en lugar de inventar (problema de "fuera de dominio" visto en la S13_2).
  - Si la pregunta nombra un gobierno ("plan de Duque"), solo se busca en ese gobierno.
"""
import json
import re

import numpy as np

from . import config
from .recuperacion import buscar_denso, detectar_periodos
from .segmentacion import separar_oraciones

SIN_EVIDENCIA = ("No encontré evidencia suficiente en los Planes Nacionales de Desarrollo para responder esa "
                 "pregunta. Prueba a reformularla o a nombrar un tema o un gobierno concretos.")
PEDIR_DETALLE = "¿Puedes darme más detalle? Por ejemplo, el tema y el gobierno o periodo que te interesan."

PLANTILLA = """Eres un asistente que responde preguntas sobre los Planes Nacionales de Desarrollo (PND) de Colombia.

A partir del CONTEXTO, responde a la PREGUNTA.
- Usa SOLO la información del contexto y cita la fuente con su número entre corchetes, por ejemplo [2].
- No repitas el contexto ni la pregunta.
- Si el contexto no es suficiente, responde: "No hay evidencia suficiente en los PND consultados."

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA (en español, breve y precisa):
"""


# ------------------------------------------------------------------ umbral
def guardar_umbral(valor):
    """Guarda el umbral calibrado en config.RESULTADOS_DIR/umbral.json."""
    (config.RESULTADOS_DIR / "umbral.json").write_text(json.dumps({"umbral": float(valor)}))


def cargar_umbral():
    """Umbral calibrado si existe; si no, el provisional de config.py."""
    ruta = config.RESULTADOS_DIR / "umbral.json"
    return json.loads(ruta.read_text())["umbral"] if ruta.exists() else config.UMBRAL_SIMILITUD


# ------------------------------------------------------------------ respuesta extractiva (S13_2)
def respuesta_extractiva(pregunta, evidencias, embeddings, n_frases=3):
    """Las frases de los párrafos recuperados más parecidas a la pregunta, cada una con su cita [n]."""
    frases = []
    for n, ev in enumerate(evidencias, 1):
        frases += [(f, n) for f in separar_oraciones(ev["texto"]) if len(f.split()) >= 6]
    if not frases:
        return SIN_EVIDENCIA
    q = np.array(embeddings.embed_query(pregunta))
    F = np.array(embeddings.embed_documents([f for f, _ in frases]))
    mejores = np.argsort(F @ q)[::-1][:n_frases]
    return "\n".join(f"• {frases[j][0]} [{frases[j][1]}]" for j in sorted(mejores, key=lambda j: frases[j][1]))


# ------------------------------------------------------------------ cadena con LLM (S14_1)
def crear_llm(modelo_id=None):
    """TinyLlama dentro de HuggingFacePipeline, igual que en la clase S14_1. Necesita GPU para ir con soltura."""
    import torch
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    modelo_id = modelo_id or config.MODELO_LLM
    tokenizer = AutoTokenizer.from_pretrained(modelo_id)
    modelo = AutoModelForCausalLM.from_pretrained(
        modelo_id,
        device_map="auto",                                                   # GPU si hay, si no CPU
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    generador = pipeline("text-generation", model=modelo, tokenizer=tokenizer, max_new_tokens=256,
                         temperature=0.3, do_sample=True, return_full_text=False)  # solo la respuesta, sin el prompt
    return HuggingFacePipeline(pipeline=generador)


def formatear_contexto(docs):
    """Numera los párrafos [1], [2]... con su cita, para que el modelo pueda citarlos (el format_docs de la clase)."""
    return "\n\n".join(
        f"[{i}] (PND {d.metadata['periodo']}, p. {d.metadata['pagina']})\n{d.page_content}"
        for i, d in enumerate(docs, 1))


def crear_cadena(vs, llm, k=config.K, periodos=None):
    """La cadena RAG de la clase: retriever | formatear -> prompt -> llm."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    search_kwargs = {"k": k}
    if periodos:
        search_kwargs["filter"] = lambda meta: meta["periodo"] in periodos
        search_kwargs["fetch_k"] = vs.index.ntotal                          # ver nota en buscar_denso
    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    prompt = PromptTemplate(input_variables=["context", "question"], template=PLANTILLA)
    return {"context": retriever | formatear_contexto, "question": RunnablePassthrough()} | prompt | llm


def citas_validas(texto, n_fuentes):
    """¿Toda cita [n] de la respuesta apunta a una fuente que existe?"""
    citas = {int(c) for c in re.findall(r"\[(\d+)\]", texto)}
    return bool(citas) and all(1 <= c <= n_fuentes for c in citas)


# ------------------------------------------------------------------ función principal
def responder(pregunta, vs, embeddings, llm=None, k=config.K, umbral=None, usar_periodo=True):
    """Responde una pregunta. Sin `llm` la respuesta es extractiva; con `llm`, generada con la cadena de LangChain.

    Devuelve un diccionario: respuesta, evidencias (lista de párrafos con su cita), similitud
    del mejor párrafo y `suficiente` (False si no se superó el umbral).
    """
    umbral = cargar_umbral() if umbral is None else umbral
    if len(re.findall(r"\w+", pregunta)) < 3:
        return {"respuesta": PEDIR_DETALLE, "evidencias": [], "similitud": 0.0, "suficiente": False}

    periodos = detectar_periodos(pregunta) if usar_periodo else None
    evidencias = buscar_denso(vs, pregunta, k, periodos)
    similitud = max((e["similitud"] for e in evidencias), default=0.0)
    if not evidencias or similitud < umbral:
        return {"respuesta": SIN_EVIDENCIA, "evidencias": evidencias, "similitud": similitud, "suficiente": False}

    if llm is None:
        texto = respuesta_extractiva(pregunta, evidencias, embeddings)
    else:
        texto = crear_cadena(vs, llm, k, periodos).invoke(pregunta).strip()
        if not citas_validas(texto, len(evidencias)) and "No hay evidencia" not in texto:
            texto += "\n\n(Aviso: el modelo no citó fuentes válidas; revisa las evidencias.)"
    return {"respuesta": texto, "evidencias": evidencias, "similitud": similitud, "suficiente": True}


def texto_chat(resp):
    """La respuesta más la lista de fuentes, lista para mostrar en el chat."""
    if not resp["suficiente"]:
        return resp["respuesta"]
    fuentes = "\n".join(f"{i}. PND {e['periodo']} ({e['presidente']}), {e['documento']}, p. {e['pagina']} "
                        f"(similitud {e['similitud']:.2f})" for i, e in enumerate(resp["evidencias"], 1))
    return f"{resp['respuesta']}\n\nFuentes:\n{fuentes}"
