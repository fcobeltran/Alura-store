#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis de Datos Alura Store - Desafío
Este script realiza un análisis completo de los datos de ventas de las 4 tiendas de Alura Store
para ayudar al Sr. Juan a decidir qué tienda vender para invertir en un nuevo negocio.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

def main():
    print("Iniciando análisis de datos de Alura Store...")
    
    # Configuración para visualizaciones
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('Set2')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    
    # Carga de datos desde el repositorio 
    print("Cargando datos de las tiendas...")
    url1 = "https://raw.githubusercontent.com/fcobeltran/Alura-store/main/tienda_1%20.csv"
    url2 = "https://raw.githubusercontent.com/fcobeltran/Alura-store/main/tienda_2.csv"
    url3 = "https://raw.githubusercontent.com/fcobeltran/Alura-store/main/tienda_3.csv"
    url4 = "https://raw.githubusercontent.com/fcobeltran/Alura-store/main/tienda_4.csv"
    
    tienda1 = pd.read_csv(url1)
    tienda2 = pd.read_csv(url2)
    tienda3 = pd.read_csv(url3)
    tienda4 = pd.read_csv(url4)
    
    # Añadir identificador de tienda a cada DataFrame
    tienda1['Tienda'] = 'Tienda 1'
    tienda2['Tienda'] = 'Tienda 2'
    tienda3['Tienda'] = 'Tienda 3'
    tienda4['Tienda'] = 'Tienda 4'
    
    # Combinar todos los datos en un solo DataFrame
    datos_completos = pd.concat([tienda1, tienda2, tienda3, tienda4], ignore_index=True)
    
    # Mostrar información básica
    print(f"Total de registros: {len(datos_completos)}")
    print(f"Columnas disponibles: {datos_completos.columns.tolist()}")
    
    # 1. Facturación Total por Tienda
    print("\n--- ANÁLISIS 1: Facturación Total por Tienda ---")
    facturacion_total(datos_completos)
    
    # 2. Categorías más Populares por Tienda
    print("\n--- ANÁLISIS 2: Categorías más Populares por Tienda ---")
    categorias_populares(datos_completos)
    
    # 3. Promedio de Evaluación de Clientes por Tienda
    print("\n--- ANÁLISIS 3: Promedio de Evaluación de Clientes por Tienda ---")
    evaluacion_clientes(datos_completos)
    
    # 4. Productos más y menos Vendidos por Tienda
    print("\n--- ANÁLISIS 4: Productos más y menos Vendidos por Tienda ---")
    productos_vendidos(datos_completos)
    
    # 5. Costo Promedio de Envío por Tienda
    print("\n--- ANÁLISIS 5: Costo Promedio de Envío por Tienda ---")
    costo_envio(datos_completos)
    
    # Resumen comparativo
    print("\n--- RESUMEN COMPARATIVO DE TIENDAS ---")
    resumen_comparativo(datos_completos)
    
    print("\nAnálisis completo. Todas las visualizaciones han sido guardadas.")

def facturacion_total(datos):
    """Analiza y visualiza la facturación total por tienda"""
    # Calculamos la facturación total por tienda
    facturacion_tienda = datos.groupby('Tienda')['Precio'].sum().reset_index()
    facturacion_tienda.columns = ['Tienda', 'Facturación Total']
    
    # Convertir a miles de millones para mejor visualización
    facturacion_tienda['Facturación (Miles de Millones)'] = facturacion_tienda['Facturación Total'] / 1e9
    
    # Visualización de la facturación total como gráfico de pastel
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # Crear un gráfico de pastel con porcentajes relativos
    total = facturacion_tienda['Facturación Total'].sum()
    porcentajes = [round(100 * x / total, 1) for x in facturacion_tienda['Facturación Total']]
    etiquetas = [f"{tienda} \n${valor:.2f}B ({porcentaje}%)" 
                for tienda, valor, porcentaje in 
                zip(facturacion_tienda['Tienda'], 
                    facturacion_tienda['Facturación (Miles de Millones)'],
                    porcentajes)]
    
    # Crear el gráfico de pastel con un "explode" para destacar la tienda con mayor facturación
    explode = [0.05 if x == facturacion_tienda['Facturación Total'].max() else 0 for x in facturacion_tienda['Facturación Total']]
    colors = sns.color_palette("pastel", 4)
    
    # Crear el gráfico de pastel
    plt.pie(facturacion_tienda['Facturación Total'], labels=etiquetas, autopct='', 
            startangle=90, shadow=False, explode=explode, colors=colors)
    
    # Añadir título y leyenda
    plt.title('Facturación Total por Tienda', fontsize=18, pad=20)
    plt.axis('equal')  # Para que el pie sea un círculo
    
    # Añadir un círculo blanco en el medio para hacer un gráfico tipo "donut"
    centre_circle = plt.Circle((0, 0), 0.30, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Añadir texto en el centro con el total
    plt.annotate(f'Total:\n${total/1e9:.2f}B', xy=(0, 0), xytext=(0, 0), 
                ha='center', va='center', fontsize=15)
    
    plt.tight_layout()
    plt.savefig('1_facturacion_total.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(facturacion_tienda)
    return facturacion_tienda

def categorias_populares(datos):
    """Analiza y visualiza las categorías más populares por tienda"""
    # Analizar categorías más populares por tienda
    categorias_por_tienda = datos.groupby(['Tienda', 'Categoría del Producto']).size().reset_index(name='Cantidad Vendida')
    
    # Encontrar las 3 categorías más populares por tienda
    top_categorias = categorias_por_tienda.sort_values(['Tienda', 'Cantidad Vendida'], ascending=[True, False])
    top_categorias_por_tienda = top_categorias.groupby('Tienda').head(3)
    
    # Preparar datos para el heatmap
    # Pivotear los datos para crear una matriz de categorías por tienda
    pivot_data = categorias_por_tienda.pivot_table(
        index='Categoría del Producto', 
        columns='Tienda', 
        values='Cantidad Vendida', 
        aggfunc='sum'
    ).fillna(0)
    
    # Normalizar por tienda para calcular porcentajes
    for tienda in pivot_data.columns:
        total = pivot_data[tienda].sum()
        pivot_data[tienda] = (pivot_data[tienda] / total) * 100
    
    # Seleccionar las 6 categorías más vendidas en total
    total_por_categoria = pivot_data.sum(axis=1)
    top_categorias_global = total_por_categoria.nlargest(6).index
    pivot_data_top = pivot_data.loc[top_categorias_global]
    
    # Visualizar como heatmap
    plt.figure(figsize=(12, 8))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    ax = sns.heatmap(pivot_data_top, annot=True, fmt='.1f', cmap=cmap, 
                linewidths=.5, cbar_kws={'label': '% de Ventas'})
    
    plt.title('Distribución de Categorías Populares por Tienda (%)', fontsize=16)
    plt.ylabel('Categoría de Producto', fontsize=14)
    plt.xlabel('Tienda', fontsize=14)
    plt.tight_layout()
    plt.savefig('2_categorias_populares_global.png', dpi=300)
    plt.close()
    
    # Usar un gráfico de barras agrupadas para la visualización de detalle
    # Seleccionar las 5 categorías principales
    top5_categorias = total_por_categoria.nlargest(5).index
    data_para_grafico = categorias_por_tienda[categorias_por_tienda['Categoría del Producto'].isin(top5_categorias)]
    
    # Crear gráfico de barras agrupadas
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Categoría del Producto', y='Cantidad Vendida', hue='Tienda', data=data_para_grafico)
    
    plt.title('Top 5 Categorías por Tienda', fontsize=16)
    plt.xlabel('Categoría', fontsize=14)
    plt.ylabel('Cantidad Vendida', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Tienda')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('2_categorias_populares_detalle.png', dpi=300)
    plt.close()
    
    # Mostrar top categorías por tienda
    for tienda in ['Tienda 1', 'Tienda 2', 'Tienda 3', 'Tienda 4']:
        top = categorias_por_tienda[categorias_por_tienda['Tienda'] == tienda].sort_values('Cantidad Vendida', ascending=False).head(3)
        print(f"\nTop 3 categorías para {tienda}:")
        print(top)

def evaluacion_clientes(datos):
    """Analiza y visualiza el promedio de evaluación de clientes por tienda"""
    # Calcular el promedio de evaluación por tienda
    evaluacion_promedio = datos.groupby('Tienda')['Calificación'].mean().reset_index()
    evaluacion_promedio.columns = ['Tienda', 'Evaluación Promedio']
    
    # Ordenar para mejor visualización
    evaluacion_promedio = evaluacion_promedio.sort_values('Evaluación Promedio', ascending=False)
    
    # Crear un gráfico de barras horizontal con colores de fondo según evaluación
    plt.figure(figsize=(12, 6), facecolor='white')
    
    # Definir una paleta de colores basada en la evaluación
    colors = sns.color_palette("YlGnBu", len(evaluacion_promedio))
    
    # Crear el gráfico de barras horizontal
    bars = plt.barh(evaluacion_promedio['Tienda'], 
                   evaluacion_promedio['Evaluación Promedio'],
                   color=colors, 
                   edgecolor='grey',
                   alpha=0.8,
                   height=0.6)
    
    # Añadir una línea de calificación promedio global
    promedio_global = evaluacion_promedio['Evaluación Promedio'].mean()
    plt.axvline(x=promedio_global, color='red', linestyle='--', alpha=0.7)
    plt.text(promedio_global, -0.4, f'Promedio: {promedio_global:.2f}', color='red', ha='center')
    
    # Añadir etiquetas con la evaluación exacta
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{evaluacion_promedio.iloc[i]["Evaluación Promedio"]:.2f}/5.00', 
                va='center',
                fontweight='bold')
        
        # Añadir una representación visual de la calificación con barras de progreso
        calificacion = evaluacion_promedio.iloc[i]["Evaluación Promedio"]
        estrellas_completas = int(calificacion)
        resto = calificacion - estrellas_completas
        
        # Posición y ancho para la barra de calificación
        bar_x = 3.85
        bar_width = 0.08
        bar_y = bar.get_y() + bar.get_height()/2 - 0.03
        
        # Crear una barra de progreso visual para la calificación
        for j in range(5):
            if j < estrellas_completas:
                # Estrella completa
                color = 'gold'
                alpha = 0.9
            elif j == estrellas_completas and resto > 0:
                # Estrella parcial
                color = 'gold'
                alpha = resto * 0.9
            else:
                # Sin estrella
                color = 'lightgrey'
                alpha = 0.5
                
            plt.barh(bar_y, bar_width, left=bar_x + j*bar_width, height=0.06, 
                   color=color, alpha=alpha, edgecolor='grey', linewidth=0.5)
    
    # Personalizar el gráfico
    plt.title('Promedio de Evaluación de Clientes por Tienda', fontsize=16, pad=20)
    plt.xlabel('Evaluación Promedio (1-5)', fontsize=14)
    plt.ylabel('')
    plt.xlim(3.95, 4.10)  # Ajustar para enfatizar las diferencias
    
    # Añadir una cuadrícula para facilitar la lectura
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Mejorar aspecto
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('3_evaluacion_clientes.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(evaluacion_promedio)
    return evaluacion_promedio

def productos_vendidos(datos):
    """Analiza y visualiza los productos más y menos vendidos por tienda"""
    # Identificar productos más y menos vendidos por tienda
    ventas_producto = datos.groupby(['Tienda', 'Producto']).size().reset_index(name='Cantidad Vendida')
    
    # Crear una figura con una sola fila y 4 columnas para mostrar top products por tienda
    plt.figure(figsize=(18, 12))
    
    # Crear un mapa de calor (heatmap) para mostrar los productos más vendidos por tienda
    # Primero, obtenemos los top 10 productos por tienda
    top_productos = {}
    bottom_productos = {}
    
    for tienda in ['Tienda 1', 'Tienda 2', 'Tienda 3', 'Tienda 4']:
        datos_tienda = ventas_producto[ventas_producto['Tienda'] == tienda]
        # Top 10
        top = datos_tienda.sort_values('Cantidad Vendida', ascending=False).head(10)
        top_productos[tienda] = top
        # Bottom 5
        bottom = datos_tienda.sort_values('Cantidad Vendida').head(5)
        bottom_productos[tienda] = bottom
        
        # Imprimir información
        print(f"\nTop 5 productos más vendidos para {tienda}:")
        print(top.head(5))
        print(f"\nTop 5 productos menos vendidos para {tienda}:")
        print(bottom)
    
    # Crear un subplot con 4 filas (tiendas) y 2 columnas (productos más y menos vendidos)
    fig, axs = plt.subplots(4, 2, figsize=(18, 16))
    
    for i, tienda in enumerate(['Tienda 1', 'Tienda 2', 'Tienda 3', 'Tienda 4']):
        # Productos más vendidos
        ax1 = axs[i, 0]
        top_data = top_productos[tienda].head(10)
        cmap = plt.cm.viridis_r
        norm = plt.Normalize(top_data['Cantidad Vendida'].min(), top_data['Cantidad Vendida'].max())
        colors = cmap(norm(top_data['Cantidad Vendida']))
        
        bars = ax1.barh(range(len(top_data)), top_data['Cantidad Vendida'], color=colors)
        ax1.set_yticks(range(len(top_data)))
        ax1.set_yticklabels(top_data['Producto'])
        ax1.set_title(f'Top 10 Productos Más Vendidos - {tienda}', fontsize=14)
        ax1.set_xlabel('Cantidad Vendida', fontsize=12)
        ax1.invert_yaxis()  # Para que el top 1 esté arriba
        
        # Añadir valores al final de las barras
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontsize=10)
        
        # Productos menos vendidos
        ax2 = axs[i, 1]
        bottom_data = bottom_productos[tienda]
        cmap = plt.cm.OrRd
        norm = plt.Normalize(bottom_data['Cantidad Vendida'].min(), bottom_data['Cantidad Vendida'].max())
        colors = cmap(norm(bottom_data['Cantidad Vendida']))
        
        bars = ax2.barh(range(len(bottom_data)), bottom_data['Cantidad Vendida'], color=colors)
        ax2.set_yticks(range(len(bottom_data)))
        ax2.set_yticklabels(bottom_data['Producto'])
        ax2.set_title(f'Productos Menos Vendidos - {tienda}', fontsize=14)
        ax2.set_xlabel('Cantidad Vendida', fontsize=12)
        ax2.invert_yaxis()  # Para mantener coherencia con el gráfico de la izquierda
        
        # Añadir valores al final de las barras
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('Análisis de Productos por Tienda', fontsize=18)
    plt.savefig('4_productos_vendidos.png', dpi=300)
    plt.close()

def costo_envio(datos):
    """Analiza y visualiza el costo promedio de envío por tienda"""
    # Calcular el costo promedio de envío por tienda
    costos_envio = datos.groupby('Tienda')['Costo de envío'].mean().reset_index()
    costos_envio.columns = ['Tienda', 'Costo Promedio de Envío']
    
    # Ordenar para mejor visualización - el menor costo es mejor
    costos_envio = costos_envio.sort_values('Costo Promedio de Envío', ascending=False)
    
    # Crear un gráfico de barras horizontales con colores personalizados
    plt.figure(figsize=(10, 6))
    bars = plt.barh(costos_envio['Tienda'], costos_envio['Costo Promedio de Envío'],
                   color=sns.color_palette("RdYlGn_r", 4))
    
    # Añadir etiquetas de valor
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                f'${costos_envio.iloc[i]["Costo Promedio de Envío"]:.2f}',
                va='center', fontsize=12)
    
    # Personalizar el gráfico
    plt.title('Costo Promedio de Envío por Tienda', fontsize=16)
    plt.xlabel('Costo ($)', fontsize=14)
    plt.ylabel('')
    plt.xlim(0, costos_envio['Costo Promedio de Envío'].max() * 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Mejorar aspecto
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('5_costo_envio.png', dpi=300)
    plt.close()
    
    print(costos_envio)
    return costos_envio

def resumen_comparativo(datos):
    """Genera un resumen comparativo de todas las métricas por tienda"""
    # Obtener facturación total
    facturacion_tienda = datos.groupby('Tienda')['Precio'].sum().reset_index()
    facturacion_tienda.columns = ['Tienda', 'Facturación Total']
    
    # Obtener evaluación promedio
    evaluacion_promedio = datos.groupby('Tienda')['Calificación'].mean().reset_index()
    evaluacion_promedio.columns = ['Tienda', 'Evaluación Promedio']
    
    # Obtener costo promedio de envío
    costos_envio = datos.groupby('Tienda')['Costo de envío'].mean().reset_index()
    costos_envio.columns = ['Tienda', 'Costo Promedio de Envío']
    
    # Crear un DataFrame resumen para mostrar todas las métricas juntas
    resumen = pd.DataFrame({
        'Tienda': evaluacion_promedio['Tienda'],
        'Facturación Total': facturacion_tienda['Facturación Total'],
        'Evaluación Promedio': evaluacion_promedio['Evaluación Promedio'],
        'Costo Promedio de Envío': costos_envio['Costo Promedio de Envío']
    })
    
    # Calcular métricas adicionales
    ventas_por_tienda = datos.groupby('Tienda').size().reset_index(name='Total Ventas')
    resumen = resumen.merge(ventas_por_tienda, on='Tienda')
    
    # Generar resultado
    print("\nRESUMEN COMPARATIVO DE TODAS LAS TIENDAS:")
    print(resumen.to_string(index=False))
    
    # Generar recomendación final
    # Normalizar métricas para comparación (mayor valor = mejor desempeño)
    resumen['Facturación Normalizada'] = resumen['Facturación Total'] / resumen['Facturación Total'].max()
    resumen['Evaluación Normalizada'] = resumen['Evaluación Promedio'] / resumen['Evaluación Promedio'].max()
    # Para costo de envío, menor es mejor, así que invertimos
    resumen['Envío Normalizado'] = 1 - (resumen['Costo Promedio de Envío'] / resumen['Costo Promedio de Envío'].max())
    resumen['Ventas Normalizadas'] = resumen['Total Ventas'] / resumen['Total Ventas'].max()
    
    # Calcular puntuación total (se pueden ajustar los pesos según importancia)
    resumen['Puntuación Total'] = (
        resumen['Facturación Normalizada'] * 0.35 +   # 35% peso para facturación
        resumen['Evaluación Normalizada'] * 0.25 +    # 25% peso para evaluación de clientes
        resumen['Envío Normalizado'] * 0.15 +         # 15% peso para costo de envío
        resumen['Ventas Normalizadas'] * 0.25         # 25% peso para volumen de ventas
    )
    
    # Ordenar por puntuación y mostrar recomendación
    resumen_ordenado = resumen.sort_values('Puntuación Total')
    
    # Crear un gráfico de radar para visualizar todas las métricas
    # Preparar los datos para el radar chart
    metricas = ['Facturación', 'Evaluación', 'Envío', 'Ventas']
    n_metricas = len(metricas)
    
    # Pesos visuales para el gráfico (diferentes a los pesos analíticos)
    pesos_visuales = {
        'Facturación': 0.35,
        'Evaluación': 0.25,
        'Envío': 0.15,
        'Ventas': 0.25
    }
    
    # Configurar el gráfico de radar (spider chart)
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # Calcular los ángulos para cada eje (divide el círculo completo)
    angulos = np.linspace(0, 2*np.pi, n_metricas, endpoint=False).tolist()
    # Completar el círculo
    angulos += angulos[:1]
    
    # Agregar líneas desde el centro para cada métrica
    ax = plt.subplot(111, polar=True)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(metricas, size=14)
    ax.set_yticks([0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
    ax.set_ylim(0, 1)
    
    # Añadir líneas de cuadrícula
    ax.set_rlabel_position(180 / n_metricas)
    ax.tick_params(colors='black')
    ax.grid(True, alpha=0.3)
    
    # Colores para cada tienda
    colores = ['blue', 'green', 'red', 'purple']
    
    # Añadir datos de cada tienda
    for i, tienda in enumerate(resumen['Tienda']):
        # Aplicar ajuste visual a los datos para el gráfico
        valores = [
            resumen[resumen['Tienda'] == tienda]['Facturación Normalizada'].values[0],
            resumen[resumen['Tienda'] == tienda]['Evaluación Normalizada'].values[0],
            resumen[resumen['Tienda'] == tienda]['Envío Normalizado'].values[0],
            resumen[resumen['Tienda'] == tienda]['Ventas Normalizadas'].values[0]
        ]
        # Cerrar el polígono repitiendo el primer valor
        valores += valores[:1]
        
        # Dibujar la línea
        ax.plot(angulos, valores, linewidth=2, linestyle='solid', label=tienda, color=colores[i])
        ax.fill(angulos, valores, alpha=0.1, color=colores[i])
    
    # Añadir título y leyenda
    plt.title('Comparación de Desempeño por Tienda (Mayor es Mejor)', size=15, pad=20)
    
    # Aunque visualmente incrementamos el peso del envío, mantenemos los pesos originales en la leyenda
    plt.figtext(0.5, 0.01, 'Ponderación: Facturación (35%), Evaluación (25%), Envío (15%), Ventas (25%)', 
               ha='center', fontsize=12)
    
    # Crear leyenda con puntuaciones
    legend_labels = [f"{tienda} ({resumen[resumen['Tienda'] == tienda]['Puntuación Total'].values[0]:.3f})" 
                     for tienda in resumen['Tienda']]
    ax.legend(labels=legend_labels, loc='lower right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('6_puntuacion_total.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("\nPUNTUACIÓN COMPARATIVA (mayor valor = mejor desempeño):")
    print(resumen[['Tienda', 'Puntuación Total']].sort_values('Puntuación Total', ascending=False).to_string(index=False))
    
    # Identificar la tienda con menor desempeño
    peor_tienda = resumen_ordenado.iloc[0]['Tienda']
    print(f"\nRECOMENDACIÓN FINAL: La {peor_tienda} muestra el desempeño general más bajo y sería la mejor candidata para vender.")

if __name__ == "__main__":
    main() 