# Guion de defensa — CarDetector

Duración objetivo: **10–12 minutos** + preguntas.  
Archivo PPT: `CarDetector_Defensa_Tesis.pptx` (notas del orador incluidas en cada slide).

---

## Antes de empezar (checklist)

1. Backend levantado: `python app.py` → `GET /detector/health` OK.  
2. Frontend: `streamlit run main.py`.  
3. Video demo corto listo: `test1.mp4` (y Plan B: capturas / video ya procesado).  
4. PPT en modo presentador (notas visibles en tu pantalla).

---

## Slide 1 — Portada (~20 s)

> Buenos días / buenas tardes. Soy **Lautaro Giménez Gil** y presento **CarDetector**: un sistema para **censar vehículos en video** y **reconocer matrículas argentinas**.  
> En unos diez minutos recorro el problema, la arquitectura, los resultados y, si alcanza el tiempo, una demo corta.

---

## Slide 2 — Agenda (~15 s)

> El orden es: problema y objetivos, arquitectura frontend/backend, los dos pipelines de visión, evaluación, limitaciones y cierre. Al final, demo si el tiempo lo permite.

---

## Slide 3 — Problema (~45–60 s)

> El conteo y la lectura de patentes a mano no escalan: son lentos y con error humano.  
> Hace falta automatizar el **censo por tipo** (auto, moto, bus, camión) y la **lectura de patentes AR**, en condiciones reales de día y noche.  
> El aporte de la tesis no es solo un modelo en un notebook: es un **sistema usable** — interfaz + API — reproducible y evaluable.

---

## Slide 4 — Objetivos (~45 s)

> Objetivos específicos: implementar censo en video, detectar y leer matrículas argentinas, exponer ambos flujos por API e interfaz, y evaluar con un protocolo documentado.  
> **Delimitación:** no entrené un detector SOTA desde cero ni pretendo reemplazar peajes comerciales; reutilizo modelos preentrenados de dominio y me concentro en la **integración end-to-end**.

---

## Slide 5 — Arquitectura (~60–75 s)

> Hay dos componentes:  
> - **Frontend Streamlit** para el operador.  
> - **Backend Flask** con API REST, jobs asíncronos y modelos cargados una sola vez.  
> Dos pipelines: **censo** (YOLOv5 + centroides) y **patentes** (YOLOv4-tiny JustAnotherAlpr + OCR ConvALPR).  
> El diseño es headless: el servidor no depende de ventana gráfica.

*(Si tenés diagrama en la tesis, podés proyectarlo aquí 5 segundos.)*

---

## Slide 6 — Censo (~60 s)

> Para vehículos uso **YOLOv5s COCO**, clases 2, 3, 5 y 7.  
> El **tracking por centroides** mantiene un ID entre frames para no contar el mismo vehículo muchas veces.  
> Agregué filtros de confianza, área y aspecto porque en calle aparecen falsos positivos (por ejemplo carteles).  
> Como corro en **CPU**, uso stride y downscale para que el pipeline sea usable en la demo.

---

## Slide 7 — Matrículas (~60 s)

> Las patentes argentinas tienen formato propio; por eso no usé un OCR genérico.  
> **JustAnotherAlpr** localiza la placa; **ConvALPR** lee los caracteres.  
> El pipeline admite parámetro día/noche.  
> La decisión de diseño fue **reutilizar pesos especializados**, no reentrenar desde cero.

---

## Slide 8 — Frontend (~40 s)

> Streamlit permite a un operador no técnico subir un video, elegir censo o matrículas, ver progreso y bajar CSV.  
> Los jobs asíncronos + polling evitan congelar la UI en videos largos.  
> Eso cierra el aporte “producto mínimo” de la tesis.

---

## Slide 9 — Metodología (~50 s)

> Corrí un batch sobre una docena de videos con un script reproducible.  
> Métricas: error de censo vs. referencia, y F1 de OCR con revisión.  
> Soy explícito: el GT más sólido es **test1**; en otros casos hay estimaciones orientativas.  
> Eso es honestidad metodológica, no un defecto oculto.

---

## Slide 10 — Resultados (~60–75 s)

> En **test1** con GT manual: censo **5/5 (0% error)** y F1 OCR alrededor de **0.75**.  
> En escenas densas o de noche el sistema sigue produciendo salidas, pero el OCR se degrada (placa chica, blur, iluminación).  
> Conclusión práctica: la **integración funciona**; el límite actual es calidad de placa + tracking simple.

---

## Slide 11 — Limitaciones (~40 s)

> Tracking por centroides no alcanza en escenas saturadas.  
> Jobs en memoria; latencia alta en CPU sin optimizaciones.  
> GT completo pendiente en gran parte del corpus.  
> Preferí decirlo yo antes de que lo diga el tribunal.

---

## Slide 12 — Conclusiones (~45 s)

> Entregué un sistema end-to-end operativo, con dos pipelines justificados y protocolo experimental.  
> Futuro: SORT/ByteTrack, cola Redis, más GT, y mejor aceleración.  
> El valor de la tesis es la **integración evaluable**, no un paper de un modelo nuevo.

---

## Slide 13 — Demo (~2–3 min)

Orden sugerido:

1. Mostrar health del backend (modelos OK).  
2. Subir `test1.mp4`.  
3. Correr **Censo** → totales.  
4. Correr **Matrículas** → placas.  
5. Mencionar CSV.

**Plan B:** si falla el live, abrir capturas / video ya anotado / `experiments/predictions/`.

Frase de transición:  
> Les muestro el flujo completo con un video corto ya validado.

---

## Slide 14 — Gracias

> Gracias. Quedo a disposición de las preguntas.

---

## Preguntas frecuentes (respuestas cortas)

| Pregunta | Respuesta |
|----------|-----------|
| ¿Por qué no SORT? | Suficiente para conteo en escenas moderadas; SORT/ByteTrack es trabajo futuro documentado. |
| ¿Por qué dos frameworks (PyTorch + TF/otros)? | Pipelines distintos; censo COCO vs. dominio AR. Unificar es futuro. |
| ¿GPU AMD? | PyTorch sin CUDA en esa GPU → CPU + stride/downscale. |
| ¿Métricas del resto de videos? | Orientativas / estimadas; el ancla con GT es test1. |
| ¿Tiempo real? | Near-real-time con stride; no garantizo FPS de producción en 1080p60 CPU. |
| ¿Datos personales / privacidad? | Videos de prueba / dominio de tesis; en despliegue real haría falta política de retención. |

---

## Tips de presentación

- Hablá **hacia el tribunal**, no hacia la pantalla.  
- En arquitectura y resultados: **una idea por frase**.  
- Si se traban: “vuelvo a eso en la demo / en limitaciones”.  
- No inventes números: si no está en Cap. 3 / CSV, no lo digas.
