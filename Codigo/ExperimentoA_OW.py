import numpy as np
import cupy as cp
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

# 1. VERSIÓN PYTHON PURO (La base de la ineficiencia)
def python_pure_simulation(size):
    # Generamos los datos usando listas para ser "puros"
    proyectiles = [[np.random.random() for _ in range(3)] for _ in range(size)]
    centro = [0.5, 0.5, 0.5]
    daños = []
    
    start_time = time.time()
    for p in proyectiles:
        # d = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        dist = math.sqrt((p[0]-centro[0])**2 + (p[1]-centro[1])**2 + (p[2]-centro[2])**2)
        # f(d) = e^-d * cos(10d)
        daño = math.exp(-dist) * math.cos(dist * 10)
        daños.append(daño)
    return time.time() - start_time

# 2. VERSIÓN VECTORIZADA (NumPy/CuPy)
def vector_simulation(lib, size):
    proyectiles = lib.random.random((size, 3)).astype(lib.float32)
    centro = lib.array([0.5, 0.5, 0.5], dtype=lib.float32)
    
    start_time = time.time()
    distancias = lib.sqrt(lib.sum((proyectiles - centro)**2, axis=1))
    daño = lib.exp(-distancias) * lib.cos(distancias * 10)
    
    if lib.__name__ == 'cupy':
        cp.cuda.Stream.null.synchronize()
    return time.time() - start_time

tamanos = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
resultados = []

print(f"--- EXPERIMENTO A: COMPARATIVA TOTAL DE INFRAESTRUCTURA ---")

# Estimamos Python Puro solo para el primer tamaño para no bloquear la PC
print("\n>>> Midiendo Python Puro (Carga de referencia: 100k)...")
t_pure_small = python_pure_simulation(100_000)
t_pure_est = t_pure_small * 10 # Estimación para 1M

for s in tamanos:
    print(f"\n>>> Escala: {s:,} proyectiles")
    
    # NumPy (CPU Vectorizada)
    t_cpu = np.mean([vector_simulation(np, s) for _ in range(3)])
    std_cpu = np.std([vector_simulation(np, s) for _ in range(3)])
    
    # CuPy (GPU Paralela)
    t_gpu = np.mean([vector_simulation(cp, s) for _ in range(3)])
    std_gpu = np.std([vector_simulation(cp, s) for _ in range(3)])
    
    speedup_gpu = t_cpu / t_gpu
    # El speedup vs Python puro será astronómico
    speedup_total = (t_pure_est * (s/1_000_000)) / t_gpu

    print(f"  [CPU Vec] {t_cpu:.4f}s | [GPU] {t_gpu:.4f}s")
    print(f"  ==> Speedup GPU vs NumPy: {speedup_gpu:.2f}x")
    
    resultados.append({
        'size': s, 
        'pure_est': t_pure_est * (s/1_000_000),
        'cpu_avg': t_cpu, 'cpu_std': std_cpu,
        'gpu_avg': t_gpu, 'gpu_std': std_gpu,
        'speedup': speedup_gpu
    })

# Guardar y Graficar
df = pd.DataFrame(resultados)
df.to_csv('../logs/metricas_exp_a_final.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(df['size'], df['pure_est'], 'r--', label='Python Puro (Estimado)', alpha=0.5)
plt.errorbar(df['size'], df['cpu_avg'], yerr=df['cpu_std'], label='CPU (NumPy)', fmt='o-', color='blue')
plt.errorbar(df['size'], df['gpu_avg'], yerr=df['gpu_std'], label='GPU (CuPy)', fmt='s-', color='green')
plt.yscale('log')
plt.title('Comparativa de Infraestructura: Del Bucle a la GPU')
plt.xlabel('Número de Proyectiles (Balas)')
plt.ylabel('Tiempo de Ejecución (seg) - Escala Log')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('../visualizaciones/comparativa_total_a.png')