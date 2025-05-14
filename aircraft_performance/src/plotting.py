# -*- coding: utf-8 -*-
"""
Created on Sun May  4 15:38:48 2025

@author: Rafael Trujillo
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Optional, List
from rich.console import Console

class PerformancePlotter:
    @staticmethod
    def plot_stall_speed(
        df_clperm: pd.DataFrame,
        aircraft_name: str,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Grafica velocidad perdida (altitud vs velocidad permisible).
        
        Args:
            df_clperm: DataFrame con columnas ['Altitude_km', 'V_perm_ms']
            aircraft_name: Nombre de la aeronave (para título)
            save_path: Ruta opcional para guardar la figura
            show: Si True, muestra la figura interactiva
            
        Returns:
            Objeto Figure de matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Datos
        x = df_clperm["V_perm_ms"]
        y = df_clperm["Altitude_km"]
        
        # Gráfico principal
        ax.plot(x, y, 'b-', linewidth=2, label="Velocidad de pérdida")
        ax.fill_betweenx(y, x, alpha=0.1, color='red')  # Área bajo la curva
        
        # Configuración
        ax.set_xlabel("Velocidad permisible (m/s)", fontsize=12)
        ax.set_ylabel("Altitud (km)", fontsize=12)
        ax.set_title(f"{aircraft_name} - Velocidad de pérdida", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc="upper right")
        
        # Guardar o mostrar
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            Console().print(f"[green]✓ Gráfico guardado en:[/] [cyan]{save_path}[/]")
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_cl_vs_mach(
        df_clreq: pd.DataFrame,
        aircraft_name: str,
        highlight_altitudes: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        ylabel: str = None,
        title : str = None,
    ) -> plt.Figure:
        """
        Grafica CL requerido vs Número de Mach para múltiples altitudes.

        Args:
            df_clreq: DataFrame con columnas 'Altitude_km' y 'Mach_X.XXX'.
            aircraft_name: Nombre de la aeronave (para el título).
            highlight_altitudes: Lista de altitudes a resaltar (ej: [0, 5, 10]).
            save_path: Ruta para guardar la figura (opcional).
            show: Si True, muestra la figura interactiva.

        Returns:
            Figura de matplotlib.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extraer números de Mach desde los nombres de las columnas
        mach_columns = [col for col in df_clreq.columns if col.startswith("Mach_")]
        mach_numbers = [float(col.split("_")[1]) for col in mach_columns]
        
        # Colores para las curvas (degradado de azul)
        colors = plt.cm.viridis_r(np.linspace(0.2, 1, len(df_clreq)))
        
        # Graficar cada altitud
        for idx, row in df_clreq.iterrows():
            altitude = row["Altitude_km"]
            cl_values = row[mach_columns].values
            
            # Resaltar altitudes específicas
            linewidth = 2.5 if (highlight_altitudes and altitude in highlight_altitudes) else 1.0
            linestyle = "-" if (highlight_altitudes and altitude in highlight_altitudes) else "--"
            alpha = 1.0 if (highlight_altitudes and altitude in highlight_altitudes) else 0.6
            
            ax.plot(
                mach_numbers,
                cl_values,
                label=f"{altitude} km",
                color=colors[idx],
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha
            )
        
        # Configuración del gráfico
        ax.set_xlabel("Número de Mach", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title , fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Leyenda optimizada (mostrar solo altitudes resaltadas o un subconjunto)
        if highlight_altitudes:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                [h for h, l in zip(handles, labels) if float(l.split(" ")[0]) in highlight_altitudes],
                [l for l in labels if float(l.split(" ")[0]) in highlight_altitudes],
                title="Altitudes (km)",
                loc="upper right"
            )
        else:
            ax.legend(title="Altitudes (km)", loc="upper right", ncol=2, fontsize=8)
        
        # Guardar o mostrar
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            Console().print(f"[green]✓ Gráfico guardado en:[/] [cyan]{save_path}[/]")
        
        if show:
            plt.show()
        
        return fig
    
    
    @staticmethod
    def plot_E_req_vs_E_av(
        df_energy_req: pd.DataFrame,
        df_energy_av: pd.DataFrame,
        aircraft_name: str,
        highlight_altitudes: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        ylabel: str = None,
        title : str = None,
    ) -> plt.Figure:
        """
        Grafica CL requerido vs Número de Mach para múltiples altitudes.

        Args:
            df_clreq: DataFrame con columnas 'Altitude_km' y 'Mach_X.XXX'.
            aircraft_name: Nombre de la aeronave (para el título).
            highlight_altitudes: Lista de altitudes a resaltar (ej: [0, 5, 10]).
            save_path: Ruta para guardar la figura (opcional).
            show: Si True, muestra la figura interactiva.

        Returns:
            Figura de matplotlib.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
         # Verificar que los DataFrames tienen las mismas dimensiones
        if df_energy_req.shape != df_energy_av.shape:
            raise ValueError("Los DataFrames deben tener las mismas dimensiones")
        
        # Extraer números de Mach desde los nombres de las columnas
        mach_columns = [col for col in df_energy_req.columns if col.startswith("Mach_")]
        mach_numbers = [float(col.split("_")[1]) for col in mach_columns]
        
        # Colores para las curvas (degradado de azul)
        colors    = plt.cm.viridis_r(np.linspace(0.2, 1, len(df_energy_req)))
        colors_av = plt.cm.OrRd(np.linspace(0.2, 1, len(df_energy_av)))
        
        # Graficar cada altitud para ambos DataFrames
        for idx, (req_row, av_row) in enumerate(zip(df_energy_req.iterrows(), df_energy_av.iterrows())):
            _, req_data = req_row
            _, av_data = av_row
            altitude = req_data["Altitude_km"]
            
            # Valores a graficar
            E_req_values = req_data[mach_columns].values
            E_av_values = av_data[mach_columns].values
            
            # Resaltar altitudes específicas
            linewidth = 2.5 if (highlight_altitudes and altitude in highlight_altitudes) else 1.0
            linestyle = "-" if (highlight_altitudes and altitude in highlight_altitudes) else "--"
            alpha = 1.0 if (highlight_altitudes and altitude in highlight_altitudes) else 0.6
            
            # Graficar energía requerida (líneas continuas)
            ax.plot(
                mach_numbers,
                E_req_values,
                label=f"{altitude} km (Req)" if highlight_altitudes and altitude in highlight_altitudes else "",
                color=colors[idx],
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha
            )
            
            # Graficar energía disponible (líneas punteadas)
            ax.plot(
                mach_numbers,
                E_av_values,
                label=f"{altitude} km (Av)" if highlight_altitudes and altitude in highlight_altitudes else "",
                color=colors_av[idx],
                linewidth=linewidth,
                linestyle='-.',  # Estilo diferente para distinguir
                alpha=alpha
            )
        
        # Configuración del gráfico
        ax.set_xlabel("Número de Mach", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title , fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Leyenda optimizada (mostrar solo altitudes resaltadas o un subconjunto)
        if highlight_altitudes:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                [h for h, l in zip(handles, labels) if float(l.split(" ")[0]) in highlight_altitudes],
                [l for l in labels if float(l.split(" ")[0]) in highlight_altitudes],
                title="Altitudes (km)",
                loc="upper right"
            )
        else:
            ax.legend(title="Altitudes (km)", loc="upper right", ncol=2, fontsize=8)
        
        # Guardar o mostrar
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            Console().print(f"[green]✓ Gráfico guardado en:[/] [cyan]{save_path}[/]")
        
        if show:
            plt.show()
        
        return fig
    
    #
    @staticmethod
    def plot_SEPower_vs_mach(df_SEPower: pd.DataFrame,
                             aircraft_name: str,
                             highlight_altitudes: Optional[List[float]] = None,
                             save_path: Optional[str] = None,
                             show: bool = True,
                             **kwargs) -> plt.Figure:
        """
        Grafica el excedente específico de potencia vs Mach para diferentes altitudes.
        
        Args:
            df_SEPower: DataFrame con excedente de potencia específica
            aircraft_name: Nombre de la aeronave (para título)
            highlight_altitudes: Altitudes a resaltar
            save_path: Ruta para guardar la figura
            show: Mostrar la figura
            **kwargs: Argumentos adicionales para matplotlib
            
        Returns:
            Figura de matplotlib
        """
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        # Lógica de plotting similar a plot_cl_vs_mach pero adaptada
        mach_columns = [col for col in df_SEPower.columns if col.startswith("Mach_")]
        
        for alt_idx, row in df_SEPower.iterrows():
            alt_km = row['Altitude_km']
            mach_values = [float(col.split('_')[1]) for col in mach_columns]
            sep_values = [row[col] for col in mach_columns]
            
            # Solo plotear si hay valores válidos
            if any(not np.isnan(v) for v in sep_values):
                line_style = '-' if highlight_altitudes and alt_km in highlight_altitudes else '--'
                line_width = 2 if highlight_altitudes and alt_km in highlight_altitudes else 1
                plt.plot(mach_values, sep_values, 
                        label=f"{alt_km} km", 
                        linestyle=line_style, 
                        linewidth=line_width)
        
        plt.xlabel("Número de Mach")
        plt.ylabel("Excedente de Potencia Específica [W]")
        plt.title(f"{aircraft_name} - Excedente de Potencia vs Mach")
        plt.grid(True)
        plt.legend(title="Altitud")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        
        return fig
    #
