# dnp-rag · Asistente RAG sobre los Planes Nacionales de Desarrollo (2006-2026)

Proyecto del curso de Procesamiento de Lenguaje Natural. Parte del notebook `DNP_NLP_v4.ipynb` y usa las librerías de las clases de RAG
(S13_2, S14_1 y S14_2). **Equipo:** *A. Rubiano Carlos, Barrios Alexandra*.

El sistema recibe una pregunta («¿Qué dice el plan 2018-2022 sobre el catastro multipropósito?»), busca los párrafos más parecidos en los PND de
5 gobiernos y responde **citando gobierno, documento y página**. Si no encuentra evidencia, lo dice en vez de inventar.

## El problema que corrige esta versión

En la v4, las categorías de tema salían de un **diccionario de palabras clave** y después un modelo aprendía a predecir esas mismas etiquetas. Las
exactitudes de 88,7 % y 97,7 % solo muestran que el modelo reproduce la regla; no demuestran que identifique temas reales. Hacía falta una
muestra anotada por personas y reservada para evaluar.

¿Qué hicimos?

1. **Dejamos de evaluar contra el diccionario.** Sigue en `diccionario.py` solo para explorar el corpus; no se entrena ni se mide nada contra él.
2. **Construimos un gold escrito por personas** (`data/gold/preguntas_rag.csv`): preguntas con el documento y las páginas donde está la respuesta, buscadas en el PDF.
3. **Partimos el gold en `dev` y `test`** (`data/gold/split.csv`, congelado). Con `dev` se decide; `test` se mira una sola vez.
4. **Medimos** hit@k y MRR (con intervalo de confianza), si el sistema rechaza lo que no sabe y, a mano, la calidad de las respuestas (rúbrica de dos personas).

> **Estado actual:** el gold tiene 30 preguntas (30 de dominio y  fuera de dominio) y **todavía les faltan `documento` y `paginas`**. Hasta completarlas no hay
> resultados de evaluación, y este README no reporta ninguno. Ver `data/gold/README.md`.

## Cómo funciona

```
pregunta ─► ¿nombra un gobierno? ─► buscar en el índice FAISS ─► ¿similitud ≥ umbral? ── no ─► «no hay evidencia»
            (filtra por periodo)     (embeddings multilingües)            │ sí
                                                                          ▼
                                    respuesta extractiva con citas [n]  (o redactada por un LLM, opcional)
```

## Estructura

```
dnp-rag/
├── README.md
├── requirements.txt            # dependencias (pensadas para Colab)
├── requirements-dev.txt        # + pytest, para las pruebas
├── pytest.ini
├── notebooks/DNP_RAG_v5.ipynb  # el notebook que se ejecuta en Colab
├── src/dnp_rag/
│   ├── config.py               # rutas y parámetros (todo se cambia aquí)
│   ├── descarga.py             # 1. descargar los PDF del DNP            (v4)
│   ├── extraccion.py           # 2. texto de cada página con PyMuPDF      (v4)
│   ├── segmentacion.py         # 3. párrafos, sin índices ni tablas       (v4)
│   ├── diccionario.py          #    temas por palabras clave (solo exploración, v4)
│   ├── indice.py               # 4. embeddings + FAISS                    (S14_1)
│   ├── recuperacion.py         # 5. buscar: TF-IDF (S13_2) y denso (S14_1)
│   ├── rag.py                  # 6. el RAG: umbral, respuesta, cadena LangChain
│   ├── evaluacion.py           # 7. evaluación contra el gold
│   └── app.py                  # 8. chat con Gradio                       (S14_2)
├── tests/                      # pruebas (no necesitan internet ni GPU)
├── data/gold/                  # LO ÚNICO de data/ que se sube a Git: trabajo manual del equipo
└── docs/                       # METODOLOGIA.md y PLAN_DE_TRABAJO.md
```

`data/pdfs`, `data/corpus`, `data/indice` y `data/resultados` se regeneran con el código y no se versionan.

## Cómo ejecutarlo en Colab (reproducible)

1. Sube el repositorio a GitHub y abre `notebooks/DNP_RAG_v5.ipynb` en Colab (*Archivo → Abrir cuaderno → GitHub*).
2. *Entorno de ejecución → Cambiar tipo de entorno → GPU T4*.
3. En la primera celda, cambia `REPO_URL` por la URL del repositorio.
4. Ejecuta las celdas **en orden**. La primera instala las dependencias (unos minutos) y monta Drive; ahí se guardan PDF, corpus e índice.
   - *Prueba rápida:* `PERIODOS_PRUEBA = ["2022-2026"]` en la sección 2 descarga un solo gobierno.
   - La sección 4 (embeddings) es la lenta; el índice queda guardado y no se recalcula.
5. Las secciones 6.2 a 6.4 se saltan solas hasta que el gold tenga `documento` y `paginas`.
6. Al terminar, guarda el notebook **con salidas** (*Archivo → Descargar .ipynb*) y sube `data/gold/` a Git: Colab borra `/content` al cerrar.

**Ejecución local (opcional):** `pip install -r requirements.txt`, `jupyter notebook notebooks/DNP_RAG_v5.ipynb`. Sin GPU, el cálculo de embeddings tarda mucho más.

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest                # IDEA: unas 45 pruebas, sin internet ni GPU; incluyen ejecutar el notebook con modelos de mentira
```

## Evaluación independiente

| Qué | Cómo | Dónde |
|---|---|---|
| ¿Encuentra el párrafo correcto? | hit@1/3/5, MRR e IC95 % sobre las preguntas con página esperada; se compara contra TF-IDF | notebook 6.2 y 6.4 |
| ¿Rechaza lo que no sabe? | % de preguntas «fuera» rechazadas y % de «dominio» rechazadas por error, con un umbral calibrado en `dev` | notebook 6.3 y 6.4 |
| ¿Son buenas las respuestas? | rúbrica 0-2 (fidelidad, relevancia, citas), dos personas | notebook sección 8 |

Con 20 preguntas de dominio, cada partición tiene solo 10: un 8/10 de aciertos tiene un intervalo de confianza de [0,49 – 0,94]. Es una limitación
que declaramos y que se reduce escribiendo más preguntas (ver `docs/METODOLOGIA.md`).

## Qué clase usamos para qué

| Clase | Qué tomamos |
|---|---|
| S13_2 (RAG con TF-IDF) | `TfidfVectorizer` como línea base; respuesta extractiva con citas; el problema de las preguntas fuera de dominio |
| S14_1 (RAG con LangChain) | `HuggingFaceEmbeddings`, `FAISS`, `HuggingFacePipeline` (TinyLlama), `PromptTemplate` + `RunnablePassthrough` |
| S14_2 (chatbot) | interfaz de chat con `gradio` |

**Decisiones propias:** modelo de embeddings multilingüe (`multilingual-e5-small`) en vez de `all-MiniLM-L6-v2` (entrenado sobre todo en inglés y con ventana más corta);
umbral de similitud calibrado en vez de fijo; filtro por gobierno cuando la pregunta lo nombra.

## Limitaciones

- Pocas preguntas de evaluación → intervalos de confianza anchos.
- Las preguntas las escribimos nosotros; idealmente alguien más escribiría una parte.
- Cada pregunta se responde sola (el chat no recuerda la conversación).
- TinyLlama (1.100 M de parámetros) responde regular en español; por eso la respuesta por defecto es la extractiva.
- Las tablas se detectan con una regla y algunas pueden quedar como «párrafos».
- La página que se guarda de un párrafo es la de su inicio; si la respuesta está en la hoja siguiente, cuenta como fallo (hay una opción `tolerancia=1` para medirlo).
- Las pruebas usan embeddings de mentira: verifican la lógica, no la calidad semántica del modelo real.


