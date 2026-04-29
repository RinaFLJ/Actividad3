import time
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt

def tarea_log(id_linea):
    # Carga pesada para que los núcleos trabajen
    total = sum(i * i for i in range(1_000_000))
    return id_linea

def ejecutar_prueba(executor_class, n, workers):
    tiempos = []
    for _ in range(3):
        inicio = time.time()
        if executor_class == "secuencial":
            for i in range(n): tarea_log(i)
        else:
            with executor_class(max_workers=workers) as executor:
                list(executor.map(tarea_log, range(n), chunksize=max(1, n // (workers * 2))))
        tiempos.append(time.time() - inicio)
    return np.mean(tiempos), np.std(tiempos)

if __name__ == '__main__':
    pasos = [10, 50, 100, 250, 500]
    total_nucleos = os.cpu_count()
    resultados = []

    print(f"--- EXPERIMENTO B: COMPARATIVA DE INFRAESTRUCTURA (CPU) ---")

    for n in pasos:
        print(f"\n>>> Escenario: {n} tareas pesadas")
        
        # 1. SECUENCIAL
        m_s, std_s = ejecutar_prueba("secuencial", n, 1)
        
        # 2. HILOS (Threads)
        m_t, std_t = ejecutar_prueba(ThreadPoolExecutor, n, total_nucleos)
        
        # 3. PROCESOS (Processes) - Paralelismo real
        m_p, std_p = ejecutar_prueba(ProcessPoolExecutor, n, total_nucleos)
        
        speedup = m_s / m_p
        
        print(f"  [SEC] {m_s:.4f}s | [THREAD] {m_t:.4f}s | [PROC] {m_p:.4f}s")
        print(f"  ==> Speedup Real (Procesos): {speedup:.2f}x")
        
        resultados.append({
            'tareas': n,
            'sec_avg': m_s, 'sec_std': std_s,
            'thread_avg': m_t, 'thread_std': std_t,
            'proc_avg': m_p, 'proc_std': std_p,
            'speedup': speedup
        })

    # Guardar métricas completas
    df = pd.DataFrame(resultados)
    df.to_csv('../logs/metricas_exp_b_completo.csv', index=False)

    # Gráfico de Comparación de Paradigmas
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['tareas'], df['sec_avg'], yerr=df['sec_std'], label='Secuencial', fmt='o-')
    plt.errorbar(df['tareas'], df['thread_avg'], yerr=df['thread_std'], label='Hilos (Threads)', fmt='o-')
    plt.errorbar(df['tareas'], df['proc_avg'], yerr=df['proc_std'], label='Procesos (Multi-core)', fmt='o-')
    plt.title('Comparativa de Infraestructura: Hilos vs Procesos')
    plt.xlabel('Carga de Trabajo')
    plt.ylabel('Tiempo (seg)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../visualizaciones/comparativa_paradigmas_b.png')