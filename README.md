# Análisis de Rendimiento en Infraestructuras de Procesamiento Masivo



Este repositorio contiene la suite de experimentos desarrollada para la asignatura de \*\*Infraestructura para Ciencia de Datos\*\*



El proyecto evalúa distintos paradigmas de cómputo y arquitecturas de hardware orientados a la optimización de procesamiento de datos masivos.



##  Especificaciones de la Infraestructura

Los experimentos fueron ejecutados en una estación de trabajo con las siguientes características:

**CPU:** AMD Ryzen 5 5500 (6 núcleos / 12 hilos)

**GPU:** NVIDIA GeForce GTX 1660 (6GB GDDR5)

**RAM:** 32GB DDR4 3200MHz

**OS:** Windows 11



##  Experimentos Realizados



### A. Aceleración por Hardware (GPU vs CPU)

Simulación de detección de colisiones (\*hitboxes\*) de proyectiles en un entorno masivo. Se compara el rendimiento de \*\*Python Puro\*\*, \*\*NumPy\*\* (vectorizado) y \*\*CuPy\*\* (CUDA).



## B. Paralelismo de Tareas (Multi-core)

Procesamiento paralelo de logs de servidor (formato SA-MP) utilizando `ProcessPoolExecutor`. Evaluación del impacto del \*Global Interpreter Lock\* (GIL) en hilos frente a procesos.



## C. Computación Distribuida (Big Data)

Análisis de telemetría de combate utilizando \*\*Dask DataFrame\*\* frente a \*\*Pandas\*\*. Se analiza el fenómeno del \*overhead\* de particionamiento en volúmenes de datos que residen en RAM.



## D. Concurrencia y Latencia (I/O Bound)

Simulación de un servidor de inventarios con latencia de red de 100ms. Se demuestra la eficiencia del \*\*Event Loop\*\* de `asyncio` frente al procesamiento bloqueante tradicional.



##  Cómo ejecutar

##1. Clonar el repositorio:


git clone \[https://github.com/TU\_USUARIO/TU\_REPO.git](https://github.com/TU\_USUARIO/TU\_REPO.git)



##2. Instalar dependencias:

 pip install numpy pandas cupy dask matplotlib

