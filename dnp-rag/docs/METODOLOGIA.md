# Metodología de evaluación

## 1. La observación del profesor

> «Las categorías utilizadas para entrenar y evaluar el clasificador se generan mediante diccionarios de palabras clave aplicados al mismo texto.
> Posteriormente, el modelo aprende a predecir esas etiquetas automáticas. Por ello, exactitudes como 88,7 % o 97,7 % no demuestran que el sistema
> identifique correctamente temas reales; principalmente demuestran que reproduce la regla que creó las etiquetas. Es necesaria una muestra anotada
> manualmente por personas y reservada para evaluación.»


## 2. Cómo la resolvemos

**Cambiamos el objetivo.** En lugar de clasificar párrafos en temas (donde no existe una verdad de referencia humana), construimos un buscador con respuesta y lo
evaluamos con una tarea en la que **sí** podemos tener la "verdad humana": *dada una pregunta, ¿en qué documento y página está la respuesta?*

| | v4 | v5 |
|---|---|---|
| Qué se evalúa | un clasificador de temas | un buscador con respuesta y citas |
| Verdad de referencia | etiquetas del diccionario (una regla) | **documento y páginas** que una persona encontró en el PDF |
| ¿Depende del diccionario? | sí | no (el diccionario solo se usa para explorar) |
| Datos reservados | no | `test` |

## 3. El gold: `data/gold/preguntas_rag.csv`

Columnas: `pregunta`, `tipo` (`dominio` o `fuera`), `documento`, `paginas`, `notas`.

- **dominio:** una pregunta que los PND sí responden. `documento` es una parte del nombre del PDF y `paginas` la(s) página(s) física(s) del PDF donde está la respuesta.
- **fuera:** una pregunta que los PND no tratan; el sistema debe decir que no tiene evidencia. Incluye casos fáciles (receta, capital de un país) y difíciles (sobre Colombia, pero no sobre el contenido de los planes).


## 4. Partición dev / test

`data/gold/split.csv` asigna cada pregunta a `dev` o `test`. 


- **dev:** se usa para todas las decisiones: calibrar el umbral, ver qué falla, comparar recuperadores.
- **test:** se evalúa una sola vez, al final. No se mira el detalle de sus fallos. Si después de verlo se cambia algo del sistema, se declara en el informe.

## 5. Métricas

**Recuperación** (`evaluacion.evaluar_recuperacion`), sobre las preguntas de dominio con documento y páginas:
- **hit@k:** fracción de preguntas en que alguno de los k primeros párrafos pertenece al documento y a una de las páginas esperadas.
- **MRR:** promedio de 1/posición del primer párrafo correcto (0 si no aparece).
- Se comparan **TF-IDF** (línea base, clase S13_2) y **denso** (clase S14_1), con y sin filtro por gobierno.

**«No hay evidencia»** (`evaluacion.calibrar_umbral`, `evaluar_rechazo`): el umbral de similitud se elige en `dev` maximizando la exactitud balanceada entre preguntas de dominio y de fuera.
En `test` se reporta el % de preguntas `fuera` rechazadas (debe ser alto) y el % de `dominio` rechazadas por error (debe ser bajo).

**Respuestas** (`evaluacion.crear_hoja_rubrica`): dos personas califican por separado de 0 a 2:

| Criterio | 0 | 1 | 2 |
|---|---|---|---|
| Fidelidad | afirma cosas que no están en las fuentes | mezcla lo respaldado con lo que no | todo está respaldado |
| Relevancia | no responde lo preguntado | responde en parte | responde lo preguntado |
| Citas | citas inexistentes o a otra fuente | algunas correctas | cada [n] apunta a la fuente correcta |

Con la respuesta extractiva (frases literales) la fidelidad será alta por construcción; lo que se mide ahí es sobre todo la relevancia.

## 6. Cuánta confianza dan los números

Con 50 preguntas de dominio, `dev` y `test` tienen 25 cada una. Intervalos de confianza del 95 % (Wilson):

| Aciertos | Intervalo |
|---|---|
| 40 de 50 | [0,49 – 0,94] |
| 25 de 50 | [0,24 – 0,76] |
| 80 de 100 | [0,58 – 0,92] |
| 160 de 200 | [0,65 – 0,90] |

Por eso: (a) siempre se reporta el intervalo, (b) solo se afirma que un método es mejor si la diferencia es grande, no de uno o dos puntos, y (c) si hay tiempo, lo más útil es
**escribir más preguntas** (idealmente 40 o más de dominio).

## 7. Amenazas a la validez

- **Quien escribe las preguntas conoce el sistema**, y puede redactarlas de forma que le favorezca. Mitigación: que parte de las preguntas las escriba la otra persona, y no copiar frases del texto.
- **Respuestas en varias páginas:** si la respuesta aparece en varias partes, se listan todas las páginas válidas.
- **Página de un párrafo largo:** se registra la página donde empieza; una respuesta en la hoja siguiente cuenta como fallo (se puede medir con `tolerancia=1`).
- **Umbral calibrado con pocas preguntas:** es una estimación optimista; se reporta su evaluación en `test`.
- **Cambiar la segmentación después de anotar** puede cambiar los números de página: si pasa, hay que revisar el gold.
- **Pruebas con embeddings de mentira:** verifican la lógica, no la calidad del modelo real.

## 8. Reproducibilidad

Semilla fija (`config.SEMILLA`), partición congelada en un archivo, parámetros en `config.py`, dependencias en `requirements.txt`, índice guardado con una huella del corpus
(si cambian los PDF o la segmentación, se reconstruye solo), y pruebas automáticas (`pytest`) que incluyen ejecutar el notebook completo con datos y modelos de mentira.
