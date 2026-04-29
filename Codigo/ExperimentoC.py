import pandas as pd
import dask.dataframe as dd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def generar_datos_combate_ow(n_registros):
    """Simulación de telemetría de Overwatch para evitar contenido genérico."""
    print(f"Generando {n_registros:,} registros de combate...")
    data = {
        'match_id': np.arange(n_registros),
        'hero_id': np.random.choice(['Sojourn', 'Tracer', 'Hanzo', 'Mizuki'], n_registros),
        'damage_dealt': np.random.uniform(10, 250, size=n_registros),
        'is_headshot': np.random.choice([True, False], n_registros, p=[0.2, 0.8])
    }
    return pd.DataFrame(data)

if __name__ == '__main__':
    # 10 millones de registros para estresar la RAM y los núcleos
    n_registros = 10_000_000
    df_base = generar_datos_combate_ow(n_registros)
    
    # Configuraciones de "Cajas" (Particiones de Dask)
    # Probamos con 4, 8 y 12 cajas para ver la curva de rendimiento
    configs_cajas = [4, 8, 12] 
    resultados = []

    print(f"\n--- EXPERIMENTO C: INFRAESTRUCTURA DE BIG DATA ---")

    # 1. TEST PANDAS (La "Caja Única" Secuencial)
    print(">>> Midiendo Pandas (Referencia)...")
    t_pandas_lista = []
    for _ in range(3):
        start = time.time()
        # Tarea: Calcular el promedio de daño de todos los Headshots
        _ = df_base[df_base['is_headshot'] == True]['damage_dealt'].mean()
        t_pandas_lista.append(time.time() - start)
    
    avg_pandas = np.mean(t_pandas_lista)
    std_pandas = np.std(t_pandas_lista)
    print(f"    Promedio Pandas: {avg_pandas:.4f}s")

    # 2. TEST DASK (Múltiples "Cajas" en Paralelo)
    for n_cajas in configs_cajas:
        print(f">>> Midiendo Dask con {n_cajas} cajas...")
        t_dask_lista = []
        
        # Repartimos el log de combate en las cajas
        ddf = dd.from_pandas(df_base, npartitions=n_cajas)
        
        for _ in range(3):
            start = time.time()
            # Activamos todas las cajas simultáneamente con .compute()
            _ = ddf[ddf['is_headshot'] == True]['damage_dealt'].mean().compute()
            t_dask_lista.append(time.time() - start)
            
        avg_dask = np.mean(t_dask_lista)
        std_dask = np.std(t_dask_lista)
        speedup = avg_pandas / avg_dask
        
        resultados.append({
            'cajas': n_cajas,
            'dask_avg': avg_dask,
            'dask_std': std_dask,
            'speedup': speedup
        })
        print(f"    Promedio Dask ({n_cajas} cajas): {avg_dask:.4f}s | Speedup: {speedup:.2f}x")

    # --- PROCESAMIENTO DE RESULTADOS ---
    df_res = pd.DataFrame(resultados)
    df_res.to_csv('../logs/metricas_exp_c_final.csv', index=False)

    # --- GENERACIÓN DEL GRÁFICO COMPARATIVO ---
    plt.figure(figsize=(10, 6))

    # Línea de Pandas (es constante porque siempre es una caja)
    plt.axhline(y=avg_pandas, color='red', linestyle='--', label='Pandas (1 Caja - Base)', linewidth=2)
    
    # Línea de Dask evolucionando por número de cajas
    plt.errorbar(df_res['cajas'], df_res['dask_avg'], yerr=df_res['dask_std'], 
                 fmt='o-', color='purple', label='Dask (Multi-caja)', markersize=8, capsize=5)

    # Configuración estética
    plt.title('Experimento C: Rendimiento de Infraestructura Dask vs Pandas', fontsize=14)
    plt.xlabel('Número de "Cajas" (Particiones / Núcleos utilizados)', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
    plt.xticks(configs_cajas) # Solo muestra 4, 8 y 12 en el eje X
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Anotación técnica para el informe
    plt.text(4, avg_pandas * 1.1, "Pandas es más eficiente en RAM local", color='red', fontsize=10)

    plt.savefig('../visualizaciones/comparativa_rendimiento_c.png')
    print(f"\n[OK] Gráfico guardado en: Actividad3/visualizaciones/comparativa_rendimiento_c.png")