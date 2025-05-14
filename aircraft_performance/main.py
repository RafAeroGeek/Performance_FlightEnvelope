# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:35:29 2025

@author: Rafael Trujillo
"""
import json
from src.aircraft import Aircraft
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.plotting import PerformancePlotter
from typing import Optional

def save_results(
    aircraft_name: str,
    df_clmax: pd.DataFrame,
    df_clperm: pd.DataFrame,
    df_clreq: pd.DataFrame,
    df_cdreq: pd.DataFrame,
    df_efficiency:pd.DataFrame,
    df_energy: pd.DataFrame,
    df_energy_av: pd.DataFrame,
    df_SEPower : pd.DataFrame,
    console: Console,
    analysis_case: Optional[str] = "standard"  # Nuevo parámetro para identificar el caso
) -> str:
    """
    Guarda resultados en directorio organizado por aeronave/fecha_índice.
    Ej: outputs/cessna172/2025-05-05_00/, .../2025-05-05_01/, etc.

    Args:
        analysis_case: Identificador del tipo de análisis (ej: "high_altitude", "emergency").
    """
    base_dir = Path("outputs") / aircraft_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar el próximo índice disponible para la fecha actual
    today_prefix = datetime.now().strftime("%Y-%m-%d")
    existing_dirs = [d.name for d in base_dir.glob(f"{today_prefix}_*") if d.is_dir()]
    
    if not existing_dirs:
        next_idx = 0
    else:
        last_idx = max(int(d.split("_")[-1]) for d in existing_dirs)
        next_idx = last_idx + 1
    
    output_dir = base_dir / f"{today_prefix}_{next_idx:02d}"  # Formato: 2025-05-05_01
    output_dir.mkdir()
    
    # Guardar DataFrames
    df_clmax.to_csv(output_dir / "min_mach_clmax.csv", index=False)
    df_clperm.to_csv(output_dir / "min_mach_clperm.csv", index=False)
    df_clreq.to_csv(output_dir / "cl_required.csv", index=False)
    df_cdreq.to_csv(output_dir / "cd_required.csv", index=False)
    df_efficiency.to_csv(output_dir / "cd_required.csv", index=False)
    df_energy.to_csv(output_dir / "Energy_method.csv", index=False)
    df_energy_av.to_csv(output_dir / "Energy_av.csv", index=False)
    df_SEPower.to_csv(output_dir / "SpecificExcessPoT.csv", index=False)
    
    console.print(f"\n[bold green]✓ Resultados guardados en:[/] [cyan]{output_dir.resolve()}[/]")
    return str(output_dir)

def save_metadata(output_dir: Path, aircraft: Aircraft, params: dict, analysis_case: str):
    """Guarda metadatos incluyendo el caso de análisis."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "analysis_case": analysis_case,  # Nuevo campo
        "aircraft": {
            "name": aircraft.aircraft_name,
            "engine": aircraft.engine_name,
            "mass_kg": aircraft.mass_kg,
            "wing_area_m2": aircraft.wing_area_m2
        },
        "calculation_parameters": params,
        "output_location": str(output_dir.resolve())  # Ruta absoluta para referencia
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    try:
        console = Console()
        console.print(Panel.fit("ANÁLISIS DE PERFORMANCE DE AERONAVES", style="bold blue"))
        config_dir = "config"
        print(f"Directorio de trabajo actual: {Path.cwd()}")
        print(f"Ruta completa de configuración: {(Path.cwd() / config_dir).resolve()}")
        
        # Mostrar aeronaves disponibles
        aircraft_options = [f.stem for f in (Path(config_dir)/"aircraft").glob("*.json")]
        console.print("\nAeronaves disponibles:", style="bold blue")
        for i, name in enumerate(aircraft_options, 1):
            console.print(f"{i}. {name}")
        
        # Selección de aeronave
        selection = int(console.input("\nSeleccione aeronave (número): ")) - 1
        aircraft_name = aircraft_options[selection]
        
        # Mostrar motores disponibles
        engine_options = [f.stem for f in (Path(config_dir)/"engines").glob("*.json")]
        console.print("\nMotores disponibles:", style="bold blue")
        for i, name in enumerate(engine_options, 1):
            console.print(f"{i}. {name}")
        console.print(f"{len(engine_options)+1}. Usar motor predeterminado")
        
        # Selección de motor
        engine_selection = int(console.input("\nSeleccione motor (número): ")) - 1
        engine_name = engine_options[engine_selection] if engine_selection < len(engine_options) else None
        
        # Crear instancia
        aircraft = Aircraft(
            config_dir=config_dir,
            aircraft_name=aircraft_name,
            engine_name=engine_name 
        )
        
                
        # Paso 1: Verificación por usuario con Rich
        if not aircraft.display_configuration():
            console.print("\nConfiguración no confirmada. Saliendo del programa.", style="bold red")
            return
        
        console.print("\nIniciando cálculos de performance...", style="bold green")
        
        # Paso 2: Calcular Mach mínimos
        with console.status("[bold green]Calculando velocidades mínimas...[/]"):
            df_clmax, df_clperm = aircraft.calculate_min_mach_cl()
        
        # Paso 3: Calcular CL requerido
        with console.status("[bold green]Calculando CL requerido...[/]"):
            df_clreq = aircraft.calculate_cl_required()
        # Paso 4 Calcular el CD requerido
        df_cdreq = aircraft.calculate_cd_required()
        
        # Paso 5: Obtener eficiencia aerodinámica
        df_efficiency = aircraft.calculate_aerodynamic_efficiency(df_clreq, df_cdreq)
        
        # Después de calcular df_clreq y df_cdreq:
        energy_method = aircraft.analysis_params.get("energy_method", "force")  # Valor por defecto: "force"
        
        if energy_method == "power":
            df_efficiency = aircraft.calculate_aerodynamic_efficiency(df_clreq, df_cdreq)
            df_energy = aircraft.calculate_Power_Required(df_efficiency)
            output_name = "power_required"
            # Después de calcular df_clreq:
            df_energy_av = aircraft.Power_Available(df_clreq)
        else:  # "force"
            df_energy = aircraft.calculate_Drag_Force(df_cdreq)
            output_name = "drag_force"   

            # Después de calcular df_clreq:
            df_energy_av = aircraft.Thrust_Available(df_clreq)
            
        df_SEPower = aircraft.Specific_excess_power(df_energy_av , df_energy)
        
        
        #Comenzaos a construir la envolvente de vuelo
        
        #Velocidad de desplome 
        stall_speeds = aircraft.Stall_True_Speed(df_SEPower)
        # Convertir a DataFrame para análisis
        df_stall_speeds = pd.DataFrame(
            stall_speeds, 
            columns=['Altitude_km', 'Stall_Speed_ms', 'Mach_at_stall']
        )
        
        # 2. Obtener máximos por altitud
        max_SEPower_list = aircraft.Max_Specific_excess_power(df_SEPower)
        
        # Convertir a DataFrame para mejor visualización
        df_max_SEPower = pd.DataFrame(max_SEPower_list, 
                                     columns=['Altitude_km', 'Max_SEPower', 'Mach_at_max_SEPower'])
        
        max_speeds = aircraft.Max_True_Speed(df_SEPower)
        df_max_SEPower = pd.DataFrame(max_speeds, 
                                     columns=['Altitude_km', 'Max_True_speed', 'Mach_at_max_speed'])
        
        # Mostrar resultados
        aircraft.display_results(df_clmax, df_clperm, df_clreq)
        
        # Definir el tipo de análisis
        analysis_case = aircraft.analysis_params.get("analysis_name", "standard")  # Usa "standard" como valor por defecto si no existe
        
        # Guardar resultados
        results_dir = save_results(
            aircraft_name=aircraft.aircraft_name,
            df_clmax=df_clmax,
            df_clperm=df_clperm,
            df_clreq=df_clreq,
            df_cdreq=df_cdreq,
            df_efficiency = df_efficiency,
            df_energy = df_energy,
            df_energy_av = df_energy_av,
            df_SEPower=df_SEPower,
            console=console,
            analysis_case=analysis_case  # Pasamos el caso
        )
        
        
        #Generar el grafico de los coeficientes de levantamiento requeridos
        plot_path = str(Path(results_dir) / "00_CL_vs_mach.png")
        aircraft.plot_cl_vs_mach(df_clreq=df_clreq , 
                                 highlight_altitudes=[0, 5, aircraft.analysis_params.get("max_mach", 0.22)],
                                 save_path=plot_path,
                                 show=False)
        
        # Guardar gráfico de velocidad perdida
        plot_path = str(Path(results_dir) / "01_stall_speed.png")
        aircraft.plot_stall_speed(df_clperm, save_path=plot_path)
        

        #Generar el grafico para el coeficiente de arrastre 
        plot_path = str(Path(results_dir) / "02_CD_vs_mach.png")
        aircraft.plot_CD_vs_mach(df_cdreq=df_cdreq,
                                         highlight_altitudes=[0, 5, aircraft.analysis_params["max_altitude_km"]],
                                         save_path=plot_path,
                                         show = True)
        
        #Generar el grafico para la eficiencia aerodinámica
        plot_path = str(Path(results_dir) / "03_Effaero_vs_mach.png")
        aircraft.plot_efficiency_vs_mach(df_efficiency=df_efficiency,
                                         highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                         save_path=plot_path,
                                         show = True)
        #Generar el grafico para el metodo energetico seleccionado
        if energy_method == "power":
            #potencia
            plot_path = str(Path(results_dir) / "04_Power_vs_mach.png")
            aircraft.plot_power_vs_mach(df_energy=df_energy,
                                             highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                             save_path=plot_path,
                                             show = True)
            plot_path = str(Path(results_dir) / f"05_Power_av_vs_mach_{aircraft.engine_name}.png")
            aircraft.plot_thrust_av_vs_mach(df_energy_av=df_energy_av,
                                             highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                             save_path=plot_path,
                                             show = True)
            plot_path = str(Path(results_dir) / f"06_power_req_av_vs_mach_{aircraft.engine_name}.png")
            aircraft.plot_thrust_req_av_vs_mach(df_energy_req = df_energy,
                                                df_energy_av=df_energy_av,  
                                             highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                             save_path=plot_path,
                                             show = True)
            
        else:  # "force"
            #fuerza
            plot_path = str(Path(results_dir) / "04_Drag_vs_mach.png")
            aircraft.plot_drag_vs_mach(df_energy=df_energy,
                                             highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                             save_path=plot_path,
                                             show = True)

            plot_path = str(Path(results_dir) / f"05_thrust_vs_mach_{aircraft.engine_name}.png")
            aircraft.plot_thrust_av_vs_mach(df_energy_av=df_energy_av,
                                             highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                             save_path=plot_path,
                                             show = True)
            plot_path = str(Path(results_dir) / f"06_thrust_&_drag_vs_mach_{aircraft.engine_name}.png")
            aircraft.plot_thrust_req_av_vs_mach(df_energy_req = df_energy,
                                                df_energy_av=df_energy_av,  
                                             highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                             save_path=plot_path,
                                             show = True)
            
        #Aparti de aqui los calculos son comunes
        plot_path = str(Path(results_dir) / f"07_excess_power_vs_mach_{aircraft.engine_name}.png")
        aircraft.plot_SEPower_vs_mach(df_SEPower,
                                      highlight_altitudes=[0, 5,  aircraft.analysis_params["max_altitude_km"]],
                                      save_path=plot_path,
                                      show = True)
        
        plot_path = str(Path(results_dir) / f"08_Flight_envelope_{aircraft.engine_name}.png")
        aircraft.plot_speed_envelope(df_clmax=df_clmax,
                                    max_speeds=max_speeds,
                                    save_path=plot_path,
                                    fill_alpha=0.25,
                                    figsize=(12, 7),
                                    show=True,
                                    mooth_factor=150,
                                    style='seaborn-whitegrid'
                                )
        
        # aircraft.plot_speed_envelope(
        #                                     df_clmax,
        #                                     max_speeds,
        #                                     save_path="envolvente_suave.png",
        #                                     show_original=True,
        #                                     smooth_factor=150,  # Más puntos = más suave
        #                                     style='seaborn-whitegrid'
        #                                 )

        
        save_metadata(
            output_dir=Path(results_dir),
            aircraft=aircraft,
            params=aircraft.analysis_params,
            analysis_case=analysis_case
        )
        
         # Guardar gráficos
        if hasattr(aircraft, 'plot_flight_envelope'):
            aircraft.plot_flight_envelope(
                save_path=str(Path(results_dir) / "flight_envelope.png")
            )
        
        console.print("\n[bold green]Análisis completado exitosamente![/]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {str(e)}", style="bold red")
    

if __name__ == "__main__":
    #from pathlib import Path
    #import numpy as np
    main()