"""Etapa 4: embeddings + índice FAISS con LangChain (clase S14_1)."""
import hashlib
import json

from langchain_community.vectorstores import FAISS

from . import config


def crear_embeddings():
    """HuggingFaceEmbeddings como en la clase S14_1, con un modelo multilingüe.

    - normalize_embeddings=True: vectores de norma 1, así la distancia de FAISS equivale a la similitud coseno.
    - Los modelos E5 necesitan un prefijo distinto para los textos ("passage: ") y para las preguntas ("query: ").
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.MODELO_EMBEDDINGS,
        encode_kwargs={"normalize_embeddings": True, "prompt": config.PREFIJO_DOCUMENTO},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": config.PREFIJO_PREGUNTA},
        show_progress=True,
    )


def _metadatos(df):
    """Datos que acompañan a cada párrafo: son la CITA de la respuesta (gobierno, documento, página)."""
    tema = df["tema"] if "tema" in df else [""] * len(df)
    return [{"id": int(i), "periodo": p, "presidente": pr, "documento": d, "pagina": int(pg), "tema": t}
            for i, p, pr, d, pg, t in zip(df["id"], df["periodo"], df["presidente"], df["documento"], df["pagina"], tema)]


def construir_indice(df, embeddings):
    """Crea el índice FAISS con todos los párrafos (tarda: calcula un embedding por párrafo)."""
    return FAISS.from_texts(texts=df["texto"].tolist(), embedding=embeddings, metadatas=_metadatos(df))


def _huella(df):
    """Identifica el corpus: si cambia (otros PDF o segmentación), el índice guardado ya no sirve."""
    h = hashlib.md5()
    for t in df["texto"]:
        h.update(t[:80].encode("utf-8"))
    return {"modelo": config.MODELO_EMBEDDINGS, "n": len(df), "md5": h.hexdigest()}


def cargar_o_construir(df, embeddings):
    """Reutiliza el índice guardado en config.INDICE_DIR si corresponde a este corpus; si no, lo reconstruye."""
    carpeta = config.INDICE_DIR / "faiss_pnd"
    ruta_meta = carpeta / "meta.json"
    huella = _huella(df)
    if ruta_meta.exists() and json.loads(ruta_meta.read_text()) == huella:
        # allow_dangerous_deserialization: el índice se guarda con pickle. Solo cargar índices propios.
        print("Índice cargado desde", carpeta)
        return FAISS.load_local(str(carpeta), embeddings, allow_dangerous_deserialization=True)
    print("Construyendo el índice (la primera vez tarda varios minutos)...")
    vs = construir_indice(df, embeddings)
    carpeta.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(carpeta))
    ruta_meta.write_text(json.dumps(huella))
    return vs
