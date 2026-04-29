import asyncio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- LOGICA DEL SERVIDOR (Simulación de Ping/Latencia de 100ms) ---

async def consultar_inventario_async(jugador_id):
    # La infraestructura NO se bloquea mientras espera el ping
    await asyncio.sleep(0.1) 
    return f"Jugador {jugador_id}: OK"

def consultar_inventario_sync(jugador_id):
    # La infraestructura se queda "congelada" esperando el ping
    time.sleep(0.1) 
    return f"Jugador {jugador_id}: OK"

# --- FUNCIONES DE EJECUCIÓN ---

async def run_async_test(n_solicitudes):
    start = time.time()
    tareas = [consultar_inventario_async(i) for i in range(n_solicitudes)]
    await asyncio.gather(*tareas)
    return time.time() - start

def run_sync_test(n_solicitudes):
    start = time.time()
    for i in range(n_solicitudes):
        consultar_inventario_sync(i)
    return time.time() - start

if __name__ == '__main__':
    jugadores = [10, 50, 100, 200, 500]
    resultados_d = []

    print("--- EXPERIMENTO D: GESTIÓN DE LATENCIA (ASYNCIO) ---")
    print("Escenario: Servidor procesando peticiones de inventario con 100ms de latencia.")

    for n in jugadores:
        print(f"\n>>> Simulando {n} jugadores conectados:")

        # 1. TEST SINCRÓNICO (3 repeticiones)
        t_sync_lista = []
        for _ in range(3):
            t_sync_lista.append(run_sync_test(n))
        avg_sync = np.mean(t_sync_lista)
        std_sync = np.std(t_sync_lista)
        print(f"   [SYNC]  Promedio: {avg_sync:.4f}s (+/- {std_sync:.4f}s)")

        # 2. TEST ASINCRÓNICO (3 repeticiones)
        t_async_lista = []
        for _ in range(3):
            # En Python 3.7+ se usa asyncio.run para cada prueba
            t_async_lista.append(asyncio.run(run_async_test(n)))
        avg_async = np.mean(t_async_lista)
        std_async = np.std(t_async_lista)
        print(f"   [ASYNC] Promedio: {avg_async:.4f}s (+/- {std_async:.4f}s)")

        speedup = avg_sync / avg_async
        resultados_d.append({
            'jugadores': n,
            'sync_avg': avg_sync, 'sync_std': std_sync,
            'async_avg': avg_async, 'async_std': std_async,
            'speedup': speedup
        })

    # Guardar resultados
    df_d = pd.DataFrame(resultados_d)
    df_d.to_csv('../logs/metricas_exp_d_final.csv', index=False)

    # --- GENERACIÓN DEL GRÁFICO DE TIEMPOS (COMPARATIVA DIRECTA) ---
    plt.figure(figsize=(10, 6))

    # Línea Sincrónica (Crecimiento lineal masivo)
    plt.errorbar(df_d['jugadores'], df_d['sync_avg'], yerr=df_d['sync_std'], 
                 fmt='o-', color='red', label='Servidor Sincrónico (Bloqueante)', capsize=5)
    
    # Línea Asincrónica (Casi plana)
    plt.errorbar(df_d['jugadores'], df_d['async_avg'], yerr=df_d['async_std'], 
                 fmt='s-', color='blue', label='Servidor Asincrónico (Event Loop)', capsize=5)

    plt.title('Experimento D: Latencia de Red en Servidor de Juego', fontsize=14)
    plt.xlabel('Número de Jugadores (Peticiones Simultáneas)', fontsize=12)
    plt.ylabel('Tiempo Total de Respuesta (segundos)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Anotación para el informe
    plt.annotate(f'Mejora de {df_d["speedup"].iloc[-1]:.0f}x', 
                 xy=(500, df_d['async_avg'].iloc[-1]), 
                 xytext=(350, 15),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig('../visualizaciones/comparativa_latencia_d.png')
    print(f"\n[OK] Gráfico guardado en: Actividad3/visualizaciones/comparativa_latencia_d.png")