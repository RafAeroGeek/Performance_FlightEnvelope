# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:33:30 2025

@author: Rafael Trujillo
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.box import SIMPLE_HEAVY
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline, LinearNDInterpolator
from .params_manager import ParametersManager
from src.plotting import PerformancePlotter  # Añadir al inicio
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import scipy.interpolate as inter
from matplotlib.ticker import AutoMinorLocator

@dataclass
class Battery:
    capacity_mAh: float
    cells: int
    nominal_voltage: float

    @property
    def energy_Wh(self) -> float:
        return (self.capacity_mAh * self.cells * self.nominal_voltage) / 1000

@dataclass
class MaxLiftData:
    mach_numbers: List[float]
    cl_values: List[float]
    stall_angles_deg: List[float]

@dataclass
class DragData:
    mach_numbers: List[float]
    k_polar: List[float]
    altitudes_km: List[float]
    cd0_matrix: List[List[float]]

@dataclass
class Aerodynamics:
    max_lift: MaxLiftData
    drag: DragData

@dataclass
class Engine(ABC):
    name: str
    max_power_watts: float
    mechanical_efficiency: float

    @abstractmethod
    def get_performance(self, mach: float, alt_km: float) -> float:
        """Retorna empuje o potencia disponible según el tipo de motor."""
        pass
@dataclass
class TurbofanEngine(Engine):
    mach_numbers: List[float]
    altitudes_km: List[float]
    thrust_matrix: List[List[float]]  # Matriz de empuje (Newtons)
    _thrust_model: Any = None  # Interpolador (se inicializa luego)

    def __post_init__(self):
        thrust_array = np.array(self.thrust_matrix)  # Convertir a NumPy
        self._thrust_model = RectBivariateSpline(
            self.mach_numbers,
            self.altitudes_km,
            thrust_array.T,
            kx=1, ky=1
        )

    def get_performance(self, mach: float, alt_km: float) -> float:
        """Retorna empuje en Newtons."""
        return float(self._thrust_model(mach, alt_km, grid=False))
@dataclass
class PistonEngine(Engine):
        name: str
        operating_rpm: int
        mechanical_efficiency: float  # Eficiencia mecánica del motor
        max_power_watts: float  # Potencia máxima al nivel del mar
        power_altitude_factor: List[Tuple[float, float]]  # [(altitud_km, factor)]
        propeller_name: str
        mach_numbers: List[float]
        prop_efficiency: List[float]
        _propeller_model: Any = None  # Modelo de interpolación
        _altitude_model: Any = None  # Modelo de interpolación para altitud

        def __post_init__(self):
            # Crear modelo de interpolación para la hélice
            self._propeller_model = inter.InterpolatedUnivariateSpline(
                np.array(self.mach_numbers),
                np.array(self.prop_efficiency),
                k=2, ext='zeros'
            )
            
            # Crear modelo de interpolación para factor de altitud
            altitudes = np.array([x[0] for x in self.power_altitude_factor])
            factors = np.array([x[1] for x in self.power_altitude_factor])
            self._altitude_model = inter.InterpolatedUnivariateSpline(
                altitudes,
                factors,
                k=1, ext='const'
            )
            #

        def get_performance(self, mach: float, alt_km: float) -> float:
            """Calcula la potencia disponible considerando:
            - Pérdidas por altitud
            - Eficiencia mecánica del motor
            - Eficiencia de la hélice
            """
            # Factor de corrección por altitud
            alt_factor = float(self._altitude_model(alt_km))
            
            # Eficiencia de la hélice para el Mach actual
            prop_efficiency = float(self._propeller_model(mach))
            
            # Potencia disponible (W)
            return self.max_power_watts * self.mechanical_efficiency * alt_factor * prop_efficiency
    
def create_engine(engine_data: dict) -> Engine:
    """Factory centralizado que valida campos obligatorios."""
    required_fields = {
        "turbofan": ["name", "Thrust", "mach_numbers", "altitudes_km"],
        "piston": ["name", "max_power_watts", "operating_rpm", "mechanical_efficiency","power_altitude_factor","propeller_name","mach_numbers","prop_efficiency"]
    }
    
    # Determinar tipo de motor
    if "Thrust" in engine_data:  # Turbofán
        missing = set(required_fields["turbofan"]) - set(engine_data.keys())
        if missing:
            raise ValueError(f"Faltan campos para turbofán: {missing}")
        
        return TurbofanEngine(
            name=engine_data["name"],
            max_power_watts=engine_data.get("max_power_watts", 0),
            mechanical_efficiency=engine_data.get("mechanical_efficiency", 1.0),
            mach_numbers=engine_data["mach_numbers"],
            altitudes_km=engine_data["altitudes_km"],
            thrust_matrix=engine_data["Thrust"]
        )
    else:  # Pistón
        missing = set(required_fields["piston"]) - set(engine_data.keys())
        if missing:
            raise ValueError(f"Faltan campos para pistón: {missing}")
        
        return PistonEngine(
            name=engine_data["name"],
            operating_rpm=engine_data["operating_rpm"],
            mechanical_efficiency=engine_data["mechanical_efficiency"],
            max_power_watts=engine_data["max_power_watts"],  # <-- Asegúrate que existe
            power_altitude_factor=engine_data["power_altitude_factor"],
            propeller_name=engine_data["propeller_name"],
            mach_numbers=engine_data["mach_numbers"],  # <-- Corregí de "mach" a "mach_numbers"
            prop_efficiency=engine_data["prop_efficiency"]
        )        
    
class Aircraft:
    def __init__(self, config_dir: str, aircraft_name: str = "h2subscale", 
                 engine_name: Optional[str] = None, params_name: str = "flight_analysis"):
        """
        Args:
            config_dir: Directorio base de configuraciones
            aircraft_name: Nombre de la aeronave (debe coincidir con archivos JSON)
            engine_name: Nombre específico del motor (opcional)
            params_name: Nombre del conjunto de parámetros a cargar
        """
        self.config_dir = Path(config_dir)
        self.aircraft_name = aircraft_name
        self.engine_name = engine_name
        self._load_configurations()
        self.params_manager = ParametersManager(config_dir)
        self.analysis_params = self.params_manager.load_parameters(aircraft_name, params_name)
        valid_energy_methods = ["power", "force"]
        if self.analysis_params.get("energy_method") not in valid_energy_methods:
            raise ValueError(
            f"Método de energía inválido. Debe ser uno de: {valid_energy_methods}. "
            f"Valor actual: {self.analysis_params.get('energy_method')}"
        )
        self._initialize_aerodynamic_models()  # Nuevo método para interpoladores
        self._initialize_engine_model()  # Nuevo método para el motor
        
    def _initialize_aerodynamic_models(self):
            """Inicializa interpoladores para Kpolar y CD0 basados en datos aerodinámicos."""
            drag_data = self.aerodynamics.drag
            
            # Interpolador para Kpolar (Mach -> Kpolar)
            self._k_polar_model = interp1d(
                drag_data.mach_numbers,
                drag_data.k_polar,
                kind='linear',
                fill_value='extrapolate'
            )
            
            # Interpolador 2D para CD0 (Mach, Altitud -> CD0)
            # Crear malla de puntos para interpolación 2D
            mach_grid, alt_grid = np.meshgrid(drag_data.mach_numbers, drag_data.altitudes_km)
            
            self._cd0_model = RectBivariateSpline(
                drag_data.mach_numbers,  # Eje X (Mach)
                drag_data.altitudes_km,  # Eje Y (Altitud)
                np.array(drag_data.cd0_matrix).T,  # Matriz (Mach x Altitud)
                kx=1,  # Grado 1 (lineal)
                ky=1
            )
            
    
            
    def _initialize_engine_model(self):
        """Inicializa el modelo del motor usando el factory."""
        engine_path = self.config_dir / "engines" / f"{self.engine_name}.json"
        engine_data = self._load_json(engine_path)
        
        #Validar que el tipo de motor esta definido
        if "type_eng" not in engine_data:
            raise ValueError(f"Engine type not specified in {engine_path}")
            
        self.engine_type = engine_data["type_eng"]
        
        try:
            self.engine = create_engine(engine_data)
        except ValueError as e:
            available = [f.stem for f in (self.config_dir / "engines").glob("*.json")]
            raise ValueError(
                f"Error en configuración del motor '{self.engine_name}': {str(e)}\n"
                f"Motores disponibles: {', '.join(available)}"
            )
        
        # Mantener compatibilidad con código existente
        if isinstance(self.engine, TurbofanEngine):
            self._thrust_model = self.engine._thrust_model
        elif isinstance(self.engine, PistonEngine):
            # Para motores de pistón, exponemos los modelos si son necesarios
            self._propeller_model = self.engine._propeller_model
            self._altitude_factor_model = self.engine._altitude_model
        #
            
    def _load_configurations(self):
        """Carga configuraciones desde archivos JSON"""
        # Verificar existencia de archivos
        aircraft_path = self.config_dir / "aircraft" / f"{self.aircraft_name}.json"
        aero_path = self.config_dir / "aerodynamics" / f"{self.aircraft_name}.json"
        
        if not aircraft_path.exists():
            available = [f.stem for f in (self.config_dir / "aircraft").glob("*.json")]
            raise FileNotFoundError(
                f"Configuración de aeronave '{self.aircraft_name}' no encontrada. "
                f"Disponibles: {', '.join(available)}"
            )
        
        
        # Cargar datos principales
        aircraft_data = self._load_json(aircraft_path)
        aero_data = self._load_json(aero_path)
        
        # Asignar parámetros básicos
        self.name = aircraft_data["name"]
        self.mass_kg = aircraft_data["mass_kg"]
        self.wing_area_m2 = aircraft_data["wing_area_m2"]
        self.load_factor = aircraft_data.get("load_factor", 3.0)
        
        # Configuración de batería
        self.battery = Battery(**aircraft_data["battery"])
        
        
        
        # Cargar datos de arrastre desde el JSON
        drag_data = DragData(
            mach_numbers=aero_data["drag"]["mach_numbers"],
            k_polar=aero_data["drag"]["k_polar"],
            altitudes_km=aero_data["drag"]["altitudes_km"],
            cd0_matrix=aero_data["drag"]["cd0_matrix"]
        )
        
        # Configuración aerodinámica
        self.aerodynamics = Aerodynamics(
            max_lift=MaxLiftData(**aero_data["max_lift"]),
            drag=drag_data
        )
        
        # Cargar motor (puede ser específico o el primero encontrado)
        self._load_engine_config()
    
    def _load_engine_config(self):
        """Carga el motor usando el factory."""
        engine_path = self.config_dir / "engines" / f"{self.engine_name}.json"
        engine_data = self._load_json(engine_path)
        
        try:
            self.engine = create_engine(engine_data)
        except ValueError as e:
            available = [f.stem for f in (self.config_dir / "engines").glob("*.json")]
            raise ValueError(
                f"Error en configuración del motor '{self.engine_name}': {str(e)}\n"
                f"Motores disponibles: {', '.join(available)}"
            )
            
    @staticmethod
    def _load_json(filepath: Path) -> Dict[str, Any]:
        """Carga un archivo JSON con manejo de errores mejorado"""
        try:
            # Verificar si el path es correcto
            if not filepath.exists():
                # Diagnóstico adicional
                parent = filepath.parent
                print(f"\nDiagnóstico:")
                print(f"  - Ruta buscada: {filepath}")
                print(f"  - Directorio padre existe: {parent.exists()}")
                if parent.exists():
                    print(f"  - Archivos en directorio: {list(parent.glob('*'))}")
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
            
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido en {filepath}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error inesperado al cargar {filepath}: {str(e)}")
            
    def __str__(self):
        return (
            f"\nInformación de la Aeronave:\n"
            f"--------------------------\n"
            f"Nombre: {self.name}\n"
            f"Masa: {self.mass_kg} kg\n"
            f"Área alar: {self.wing_area_m2} m²\n"
            f"Factor de carga: {self.load_factor}\n"
            f"\nBatería:\n"
            f"- Capacidad: {self.battery.capacity_mAh} mAh\n"
            f"- Celdas: {self.battery.cells}\n"
            f"- Voltaje nominal: {self.battery.nominal_voltage} V\n"
            f"- Energía total: {self.battery.energy_Wh:.2f} Wh\n"
            f"\nMotor:\n"
            f"- Modelo: {self.engine.name}\n"
            f"- Potencia: {self.engine.max_power_watts} W\n"
            f"- RPM operación: {self.engine.operating_rpm}\n"
            f"- Eficiencia mecánica: {self.engine.mechanical_efficiency*100:.1f}%\n"
            f"\nAerodinámica:\n"
            f"- CL máximo: {self.aerodynamics.max_lift.cl_values}\n"
            f"- Ángulos de stall: {self.aerodynamics.max_lift.stall_angles_deg}°\n"
        )
    
    def pretty_print(self):
        console = Console()
        table = Table(title=f"Datos de {self.name}")
        
        table.add_column("Parámetro", style="cyan")
        table.add_column("Valor", style="green")
        
        table.add_row("Masa", f"{self.mass_kg} kg")
        table.add_row("Área alar", f"{self.wing_area_m2} m²")
        # ... añade más filas
        
        console.print(table)
        
    def display_configuration(self) -> bool:
        """Muestra la configuración en formato profesional usando Rich"""
        console = Console()
        
        # Tabla principal de configuración
        main_table = Table(title=f"Configuración de {self.name}", box=SIMPLE_HEAVY)
        main_table.add_column("Parámetro", style="cyan")
        main_table.add_column("Valor", style="green")
        
        main_table.add_row("Masa", f"{self.mass_kg:.2f} kg")
        main_table.add_row("Superficie alar", f"{self.wing_area_m2:.2f} m²")
        main_table.add_row("Factor de carga", f"{self.load_factor:.1f}")
        main_table.add_row("Energía batería", f"{self.battery.energy_Wh:.2f} Wh")
        main_table.add_row("Motor", self.engine.name)
        #main_table.add_row("Potencia motor", f"{self.engine.max_power_watts:.0f} W")
        
        # Tabla de datos aerodinámicos
        aero_table = Table(title="Datos Aerodinámicos", box=SIMPLE_HEAVY)
        aero_table.add_column("Mach", style="magenta")
        aero_table.add_column("CL máximo", style="yellow")
        aero_table.add_column("Ángulo stall", style="yellow")
        
        for mach, cl, angle in zip(self.aerodynamics.max_lift.mach_numbers,
                                  self.aerodynamics.max_lift.cl_values,
                                  self.aerodynamics.max_lift.stall_angles_deg):
            aero_table.add_row(f"{mach:.2f}", f"{cl:.2f}", f"{angle:.1f}°")
        
        #Tabla de parametros de configuración
        parameters_table = Table(title =f"Parámetros de Análisis de {self.name}", box=SIMPLE_HEAVY)
        parameters_table.add_column("Parámetro", style="cyan")
        parameters_table.add_column("Valor", style="green")
        
        parameters_table.add_row("Altitud máxima configurada", f"{self.analysis_params['max_altitude_km']:.2f} km")
        parameters_table.add_row("Incrementos del número de mach", f"{self.analysis_params['mach_step']:.5f} ")
        parameters_table.add_row("Mach mínimo propuesto:", f"{self.analysis_params['min_mach']:.5f} ")
        parameters_table.add_row("Mach máximo propuesto:", f"{self.analysis_params['max_mach']:.5f} ")
        parameters_table.add_row("Incrementos de altitud", f"{self.analysis_params['alt_step_km']:.5f} km")
        parameters_table.add_row("factor para CL permisible", f"{self.analysis_params['cl_permissible_factor']:.3f} km")
        
        
        # Mostrar todo
        console.print(Panel.fit(main_table, title="Resumen Aeronave"))
        console.print(Panel.fit(aero_table, title="Características Aerodinámicas"))
        console.print(Panel.fit(parameters_table, title="Resumen Configuración"))
        
        # Preguntar confirmación
        console.print("\n[bold]¿Los datos son correctos?[/bold]", style="blue")
        while True:
            response = console.input("[green]\[s/n][/green] > ").lower()
            if response in ['s', 'n']:
                return response == 's'
            console.print("Por favor ingrese 's' para sí o 'n' para no", style="red")
            
    def calculate_min_mach_cl(self):
        """
        Calcula los Mach mínimos para CL máximo y CL permisible
        Devuelve:
            - df_clmax: DataFrame con Mach mínimo para CL máximo
            - df_clperm: DataFrame con Mach mínimo para CL permisible (80% CLmax)
        """
        max_altitude = self.analysis_params["max_altitude_km"]
        # Modelo lineal para CL máximo
        clmax_model = interp1d(
            self.aerodynamics.max_lift.mach_numbers,
            self.aerodynamics.max_lift.cl_values,
            kind='linear',
            fill_value='extrapolate'
        )
        
        # Funciones auxiliares
        def pressure_at_altitude(alt_km: float) -> float:
            return 101325 * (1 - 0.0065 * alt_km * 1000 / 288.15) ** 5.2561
        
        def altitude_at_pressure(pressure: float) -> float:
            return (1 - (pressure/101325)**(1/5.2561)) * 288.15 / 0.0065 / 1000
        
        def true_speed(alt_km: float, mach: float) -> float:
            return 340.3 * mach * (1 - 0.0065 * alt_km * 1000 / 288.15)**0.5
        
        # Cálculo para CL máximo
        mach_clmax = 0.01
        alt_clmax = 0
        results_clmax = {'Altitude_km': [0], 'Mach_min': [], 'CL_max': [], 'V_min_ms': []}
        
        while alt_clmax < max_altitude:
            cl_max = float(clmax_model(mach_clmax))
            pressure = (self.mass_kg * 9.81) / (0.7 * cl_max * mach_clmax**2 * self.wing_area_m2)
            
            if pressure <= 101325.0:
                alt_clmax = altitude_at_pressure(pressure)
                if 0 <= alt_clmax <= max_altitude:
                    results_clmax['Altitude_km'].append(round(alt_clmax, 5))
                    results_clmax['Mach_min'].append(round(mach_clmax, 5))
                    results_clmax['CL_max'].append(round(cl_max, 5))
                    results_clmax['V_min_ms'].append(round(true_speed(alt_clmax, mach_clmax), 2))
            
            mach_clmax += 0.0001
        
        # Añadir primer valor
        first_mach = np.sqrt(self.mass_kg * 9.81 / 
                           (0.7 * pressure_at_altitude(0) * 
                            self.wing_area_m2 * results_clmax['CL_max'][0]))
        results_clmax['Mach_min'].insert(0, round(first_mach, 4))
        results_clmax['CL_max'].insert(0, round(float(clmax_model(first_mach)), 5))
        results_clmax['V_min_ms'].insert(0, round(true_speed(0, first_mach), 2))
        
        # Cálculo para CL permisible (80% CLmax)
        k_perm = 0.8
        results_clperm = {'Altitude_km': [0], 'Mach_perm': [], 'CL_perm': [], 'V_perm_ms': []}
        
        for alt, mach, cl_max in zip(results_clmax['Altitude_km'][1:], 
                                   results_clmax['Mach_min'][1:], 
                                   results_clmax['CL_max'][1:]):
            cl_perm = cl_max * k_perm
            results_clperm['Altitude_km'].append(alt)
            results_clperm['Mach_perm'].append(mach)
            results_clperm['CL_perm'].append(round(cl_perm, 5))
            results_clperm['V_perm_ms'].append(round(true_speed(alt, mach), 2))
        
        # Primer valor para CL permisible
        results_clperm['Mach_perm'].insert(0, round(np.sqrt(
            self.mass_kg * 9.81 / 
            (0.7 * pressure_at_altitude(0) * 
             self.wing_area_m2 * results_clperm['CL_perm'][0])), 4))
        results_clperm['CL_perm'].insert(0, round(results_clmax['CL_max'][0] * k_perm, 5))
        results_clperm['V_perm_ms'].insert(0, round(true_speed(0, results_clperm['Mach_perm'][0]), 2))
        
        return (
            pd.DataFrame(results_clmax),
            pd.DataFrame(results_clperm)
        )
    
    def calculate_cl_required(self, mach_step: Optional[float] = None, alt_step: Optional[float] = None) -> pd.DataFrame:
        """
        Calcula el CL requerido para diferentes altitudes y números de Mach.
        Los pasos (step) pueden especificarse o leerse desde los parámetros de configuración.
    
        Args:
            mach_step: Incremento en número de Mach (opcional). Si es None, usa el valor del JSON.
            alt_step: Incremento en altitud (km) (opcional). Si es None, usa el valor del JSON.
    
        Returns:
            DataFrame con columnas: Altitude_km, Mach_X.XXX (CL requerido para cada Mach).
        """
        # 1. Validar parámetros de entrada
        if mach_step is not None and mach_step <= 0:
            raise ValueError("mach_step debe ser positivo")
        if alt_step is not None and alt_step <= 0:
            raise ValueError("alt_step debe ser positivo")
    
        # 2. Obtener Mach mínimo del cálculo anterior
        df_clmax, _ = self.calculate_min_mach_cl()
        mach_min_actual = max(df_clmax['Mach_min'].min(), 0.01)  # Evita valores <= 0.01
    
        # 3. Cargar parámetros desde la configuración (con valores por defecto)
        config = self.analysis_params
        max_mach = config.get("max_mach", 0.22)
        used_mach_step = mach_step if mach_step is not None else config.get("mach_step", 0.05)
        used_alt_step = alt_step if alt_step is not None else config.get("alt_step_km", 0.5)
    
        # 4. Validar pasos y límites
        if used_mach_step <= 0 or used_alt_step <= 0:
            raise ValueError("Los pasos (steps) deben ser positivos")
    
        # 5. Crear rangos de Mach y altitud
        mach_numbers = np.arange(
            start=mach_min_actual,
            stop=max_mach,
            step=used_mach_step
        )
        altitudes_km = np.arange(
            start=0,
            stop=config["max_altitude_km"] + used_alt_step, #esto es para incluir la altitud maxima
            step=used_alt_step
        )
    
        # 6. Inicializar DataFrame
        results = {"Altitude_km": altitudes_km}
        for mach in mach_numbers:
            results[f"Mach_{mach:.3f}"] = np.nan  # Formato: Mach_0.125
    
        df = pd.DataFrame(results)
    
        # 7. Calcular CL requerido para cada combinación
        for mach_col in df.columns[1:]:  # Ignorar columna Altitude_km
            mach = float(mach_col.split("_")[1])
            clmax = float(
                interp1d(
                    self.aerodynamics.max_lift.mach_numbers,
                    self.aerodynamics.max_lift.cl_values,
                    kind=config["simulation_parameters"].get("interpolation_method", "linear"),
                    fill_value="extrapolate"
                )(mach)
            )
    
            for i, alt in enumerate(df["Altitude_km"]):
                pressure = 101325 * (1 - 0.0065 * alt * 1000 / 288.15) ** 5.2561
                cl_req = (self.mass_kg * 9.81) / (0.7 * pressure * mach**2 * self.wing_area_m2)
                
                if cl_req <= clmax:
                    df.at[i, mach_col] = round(cl_req, 5)  # Redondear a 5 decimales
    
        return df
    
    def display_results(self, df_clmax: pd.DataFrame, df_clperm: pd.DataFrame, df_clreq: pd.DataFrame):
        """Muestra los resultados de los cálculos en formato profesional"""
        console = Console()
        
        # Tabla para CL máximo
        clmax_table = Table(title="Mach mínimo para CL máximo", box=SIMPLE_HEAVY)
        for col in df_clmax.columns:
            clmax_table.add_column(col)
        for _, row in df_clmax.head().iterrows():
            clmax_table.add_row(*[str(round(x, 4)) if isinstance(x, (float, int)) else str(x) for x in row])
        
        # Tabla para CL permisible
        clperm_table = Table(title="Mach permisible (80% CLmax)", box=SIMPLE_HEAVY)
        for col in df_clperm.columns:
            clperm_table.add_column(col)
        for _, row in df_clperm.head().iterrows():
            clperm_table.add_row(*[str(round(x, 4)) if isinstance(x, (float, int)) else str(x) for x in row])
        
        # Mostrar todo
        console.print(Panel.fit(clmax_table))
        console.print(Panel.fit(clperm_table))
        
        # Mostrar muestra de CL requerido
        sample_table = Table(title="CL Requerido (muestra)", box=SIMPLE_HEAVY)
        sample_table.add_column("Altitud (km)")
        sample_table.add_column("Mach", style="cyan")
        sample_table.add_column("CL req", style="yellow")
        
        # Tomar algunas muestras para mostrar
        sample_data = []
        for alt_idx in range(0, len(df_clreq), len(df_clreq)//5):
            alt = df_clreq.iloc[alt_idx, 0]
            for mach_col in df_clreq.columns[1:3]:  # Mostrar solo 2 columnas Mach
                mach = float(mach_col.split('_')[1])
                cl = df_clreq.at[alt_idx, mach_col]
                if not np.isnan(cl):
                    sample_data.append((alt, mach, cl))
        
        for alt, mach, cl in sorted(sample_data, key=lambda x: (x[0], x[1])):
            sample_table.add_row(f"{alt:.1f}", f"{mach:.3f}", f"{cl:.4f}")
        
        console.print(Panel.fit(sample_table))
        
    def plot_stall_speed(self, df_clperm: pd.DataFrame, save_path: Optional[str] = None):
        """Wrapper para el plotter de velocidad de pérdida"""
        return PerformancePlotter.plot_stall_speed(
            df_clperm=df_clperm,
            aircraft_name=self.aircraft_name,
            save_path=save_path
        )
    
    def plot_cl_vs_mach(
            self, 
            df_clreq: pd.DataFrame, 
            highlight_altitudes: Optional[List[float]] = None, 
            save_path: Optional[str] = None, 
            show: bool = True
        ) -> plt.Figure:
            """Wrapper que usa el nombre de la aeronave automáticamente"""
            return PerformancePlotter.plot_cl_vs_mach(
                df_clreq=df_clreq,
                aircraft_name=self.aircraft_name,  # 🚀 Elimina redundancia
                highlight_altitudes=highlight_altitudes,
                save_path=save_path,
                show=show,
                ylabel="CL requerido",
                title=f"{self.aircraft_name} - CL vs Mach por Altitud",
            )
    

    
    def calculate_cd_required(self) -> pd.DataFrame:
        """
        Calcula el coeficiente de arrastre (CD) requerido para diferentes Mach y altitudes.
        Reutiliza modelos de CL, CD0 y Kpolar ya cargados en la clase.
    
        Returns:
            DataFrame con estructura:
            - Columnas: Altitude_km, Mach_0.100, Mach_0.150, ...
            - Valores: CD requerido para cada combinación Mach-Altitud.
        """
        # 1. Obtener CL requerido (reutilizando método existente)
        df_clreq = self.calculate_cl_required()
        
        # 2. Inicializar DataFrame para CD (misma estructura que df_clreq)
        df_cdreq = df_clreq.copy()
        mach_columns = [col for col in df_clreq.columns if col.startswith("Mach_")]
        
        # 3. Modelos existentes en la clase (asumiendo que ya están definidos)
        # Ejemplo: self.k_polar_model (interpolador de Kpolar vs Mach)
        #          self.cd0_model (interpolador 2D de CD0 vs Mach y Altitud)
        
        # 4. Calcular CD para cada combinación Mach-Altitud
        for mach_col in mach_columns:
            mach = float(mach_col.split('_')[1])
            for i, alt in enumerate(df_cdreq['Altitude_km']):
                cl = df_clreq.at[i, mach_col]
                if np.isnan(cl):  # Si CL no es válido, CD tampoco
                    df_cdreq.at[i, mach_col] = np.nan
                    continue
                    
                # Coeficientes de arrastre (ejemplo con tus fórmulas)
                #x = cl / self.max_cl_at_mach(mach)  # Reutilizar función existente
                #cdi = 0.04 * (1 - (1 - x**3)**0.25 if x <= 1 else 0.0
                cd0 = float(self._cd0_model(mach, alt))
                k_polar = float(self._k_polar_model(mach))
                
                # CD total (componentes: arrastre parásito + inducido + adicional)
                cd_total = cd0 + k_polar * cl**2 #+ cdi
                
                df_cdreq.at[i, mach_col] = cd_total
        
        return df_cdreq
    
    def validate_aerodynamic_models(self):
        """Verifica que los interpoladores estén inicializados."""
        if not hasattr(self, '_k_polar_model') or not hasattr(self, '_cd0_model'):
            raise RuntimeError("Modelos aerodinámicos no inicializados. Llame a _initialize_aerodynamic_models()")
            
    def calculate_aerodynamic_efficiency(
        self, 
        df_clreq: pd.DataFrame, 
        df_cdreq: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula la eficiencia aerodinámica (L/D) para cada combinación Mach-Altitud.
        
        Args:
            df_clreq: DataFrame con CL requerido (de calculate_cl_required()).
            df_cdreq: DataFrame con CD requerido (de calculate_cd_required()).
            
        Returns:
            DataFrame con la misma estructura que df_clreq, pero con valores L/D.
            
        Raises:
            ValueError: Si los DataFrames no tienen la misma estructura.
        """
        # Validar que los DataFrames sean compatibles
        if not df_clreq.columns.equals(df_cdreq.columns):
            raise ValueError("Los DataFrames de CL y CD deben tener las mismas columnas")
        
        # Crear DataFrame de salida (misma estructura que df_clreq)
        df_efficiency = df_clreq.copy()
        
        # Calcular L/D para cada celda (excepto la columna 'Altitude_km')
        mach_columns = [col for col in df_clreq.columns if col.startswith("Mach_")]
        for col in mach_columns:
            if (df_cdreq[mach_columns] <= 0).any().any(): #añadi esta comprobación.
                raise ValueError("CD debe ser positivo para calcular L/D")
            df_efficiency[col] = df_clreq[col] / df_cdreq[col]  # L/D = CL / CD
            
        df_efficiency.replace([np.inf, -np.inf], np.nan, inplace=True) #Evitamos valores infinitos o nan
        return df_efficiency
    
    def plot_CD_vs_mach(self, df_cdreq: pd.DataFrame, 
                                highlight_altitudes: Optional[List[float]] = None,
                                save_path: Optional[str] = None,
                                show: bool = True,
                                **kwargs):
        """Wrapper para graficar L/D vs Mach."""
        return PerformancePlotter.plot_cl_vs_mach(
            df_clreq=df_cdreq,
            aircraft_name=self.aircraft_name,
            highlight_altitudes=highlight_altitudes,
            save_path=save_path,
            show=show,
            ylabel="Coefficiente de arrastre CD",
            title=f"{self.aircraft_name} - D vs Mach por Altitud",
            **kwargs
        )
    
    def plot_efficiency_vs_mach(self, df_efficiency: pd.DataFrame, 
                                highlight_altitudes: Optional[List[float]] = None,
                                save_path: Optional[str] = None,
                                show: bool = True,
                                **kwargs):
        """Wrapper para graficar L/D vs Mach."""
        return PerformancePlotter.plot_cl_vs_mach(
            df_clreq=df_efficiency,
            aircraft_name=self.aircraft_name,
            highlight_altitudes=highlight_altitudes,
            save_path=save_path,
            show=show,
            ylabel="Eficiencia aerodinámica (L/D)",
            title=f"{self.aircraft_name} - L/D vs Mach por Altitud",
            **kwargs
        )
    
    def plot_power_vs_mach(self, df_energy: pd.DataFrame, 
                                highlight_altitudes: Optional[List[float]] = None,
                                save_path: Optional[str] = None,
                                show: bool = True,
                                **kwargs):
        """Wrapper para graficar P_req vs Mach."""
        return PerformancePlotter.plot_cl_vs_mach(
            df_clreq=df_energy,
            aircraft_name=self.aircraft_name,
            highlight_altitudes=highlight_altitudes,
            save_path=save_path,
            show=show,
            ylabel="Potencia [Watts]",
            title=f"{self.aircraft_name} - Potencia vs Mach por Altitud",
            **kwargs
        )
    
    def plot_drag_vs_mach(self, df_energy: pd.DataFrame, 
                                highlight_altitudes: Optional[List[float]] = None,
                                save_path: Optional[str] = None,
                                show: bool = True,
                                **kwargs):
        """Wrapper para graficar D vs Mach."""
        return PerformancePlotter.plot_cl_vs_mach(
            df_clreq=df_energy,
            aircraft_name=self.aircraft_name,
            highlight_altitudes=highlight_altitudes,
            save_path=save_path,
            show=show,
            ylabel="Arrastre [Newtons] ",
            title=f"{self.aircraft_name} - Arrastre vs Mach por Altitud",
            **kwargs
        )
    

    def calculate_Drag_Force(self, df_cdreq: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la fuerza de arrastre (D) para turborreactores.
        Fórmula: D = 0.7 * P_altitud * M² * S * CD
        """
        df_Drag_force = df_cdreq.copy()
        mach_columns = [col for col in df_cdreq.columns if col.startswith("Mach_")]
        
        # Vectorización del cálculo
        for col in mach_columns:
            mach = float(col.split('_')[1])
            df_Drag_force[col] = (
                0.7 
                * df_cdreq["Altitude_km"].apply(lambda alt: 101325 * (1 - 0.0065 * alt * 1000 / 288.15)**5.2561)
                * mach**2 
                * self.wing_area_m2 
                * df_cdreq[col]
            )
        
        return df_Drag_force
    
    def _true_speed(self, alt_km: float, mach: float) -> float:
        """Velocidad verdadera (m/s) para una altitud y Mach dado."""
        return 340.3 * mach * (1 - 0.0065 * alt_km * 1000 / 288.15)**0.5
    
    def calculate_Power_Required(self, df_efficiency: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la potencia requerida (P_req) para hélices.
        Fórmula: P_req = (m * g * V_true) / (L/D)
        """
        df_Power_req = df_efficiency.copy()
        mach_columns = [col for col in df_efficiency.columns if col.startswith("Mach_")]
        
        for col in mach_columns:
            mach = float(col.split('_')[1])
            df_Power_req[col] = (
                self.mass_kg * 9.81 
                * df_efficiency["Altitude_km"].apply(lambda alt: self._true_speed(alt, mach))
                / df_efficiency[col]
            )
        
        return df_Power_req
    
    def Thrust_Available(self, df_clreq: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el empuje disponible para cada combinación Mach-Altitud.
        
        Args:
            df_clreq: DataFrame de referencia con columnas Mach y altitudes (ej: el de CL requerido).
            
        Returns:
            DataFrame con misma estructura que df_clreq, pero con valores de empuje (Newtons).
        """
        df_thrust = df_clreq.copy()
        mach_columns = [col for col in df_clreq.columns if col.startswith("Mach_")]
        
        for col in mach_columns:
            mach = float(col.split('_')[1])
            df_thrust[col] = df_clreq["Altitude_km"].apply(lambda alt: float(self._thrust_model(mach, alt, grid=False)))
        
        return df_thrust
    
    def plot_thrust_av_vs_mach(self,df_energy_av: pd.DataFrame,
                                highlight_altitudes: Optional[List[float]] = None,
                                save_path: Optional[str] = None,
                                show: bool = True,
                                **kwargs):
        """Wrapper para graficar D vs Mach."""
        return PerformancePlotter.plot_cl_vs_mach(
            df_clreq=df_energy_av,
            aircraft_name=self.aircraft_name,
            highlight_altitudes=highlight_altitudes,
            save_path=save_path,
            show=show,
            ylabel="Empuje [Newtons] ",
            title=f"{self.aircraft_name} - Empuje vs Mach por Altitud",
            **kwargs
        )
    
    def plot_thrust_req_av_vs_mach(self, df_energy_req: pd.DataFrame,
                                df_energy_av: pd.DataFrame,
                                highlight_altitudes: Optional[List[float]] = None,
                                save_path: Optional[str] = None,
                                show: bool = True,
                                **kwargs):
        """Wrapper para graficar D vs Mach."""
        return PerformancePlotter.plot_E_req_vs_E_av(
            df_energy_req=df_energy_req,
            df_energy_av=df_energy_av,
            aircraft_name=self.aircraft_name,
            highlight_altitudes=highlight_altitudes,
            save_path=save_path,
            show=show,
            ylabel="Empuje [Newtons] ",
            title=f"{self.aircraft_name} - Empuje vs Empuje vs Mach por Altitud",
            **kwargs
        )
    
    #
    def plot_SEPower_vs_mach(self,
                             df_SEPower: pd.DataFrame,
                             highlight_altitudes: Optional[List[float]] = None,
                             save_path: Optional[str] = None,
                             show: bool = True,
                             **kwargs) -> plt.Figure:
        """
        Wrapper para graficar el excedente específico de potencia vs Mach.
        
        Args:
            df_SEPower: DataFrame de excedente de potencia específica
            highlight_altitudes: Lista de altitudes a resaltar
            save_path: Ruta para guardar la figura (opcional)
            show: Mostrar la figura inmediatamente
            **kwargs: Argumentos adicionales para el plotter
            
        Returns:
            Objeto Figure de matplotlib
        """
        return PerformancePlotter.plot_cl_vs_mach(
            df_clreq=df_SEPower,
            aircraft_name=self.aircraft_name,
            highlight_altitudes=highlight_altitudes,
            save_path=save_path,
            show=show,
            ylabel="Excedente de Potencia Específica [W]",
            title=f"{self.aircraft_name} - Excedente de Potencia vs Mach",
            **kwargs
        )
    #
    
    def Power_Available(self, df_clreq: pd.DataFrame) -> pd.DataFrame:
        #aqui la funcion de potencia por altitud.
        """
        Calcula la fuerza de arrastre (D) para turborreactores.
        Fórmula: D = 0.7 * P_altitud * M² * S * CD
        """
        df_power_av = df_clreq.copy()
        mach_columns = [col for col in df_clreq.columns if col.startswith("Mach_")]
        
        for col in mach_columns:
            mach = float(col.split('_')[1])
            df_power_av[col] = df_clreq["Altitude_km"].apply(
                lambda alt: self.engine.get_performance(mach, alt)
            )
        # Calcular potencia disponible en watts

        
        return df_power_av
    
    #
    def Specific_excess_power(self, df_power_av: pd.DataFrame, df_power_req: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el excedente específico de potencia (Ps) como la diferencia entre potencia disponible y requerida.
        
        Args:
            df_power_av: DataFrame con potencia disponible (de Power_Available())
            df_power_req: DataFrame con potencia requerida (de calculate_Power_Required())
            
        Returns:
            DataFrame con mismo formato que los inputs, pero con valores de Ps (potencia específica excedente).
            Los valores negativos se reemplazan por NaN ya que no tienen significado físico.
            
        Raises:
            ValueError: Si los DataFrames no tienen la misma estructura.
        """
        # Validar que los DataFrames sean compatibles
        if not df_power_av.columns.equals(df_power_req.columns):
            raise ValueError("Los DataFrames de potencia disponible y requerida deben tener las mismas columnas")
        
        # Calcular diferencia (disponible - requerida)
        df_SEPower = df_power_av.copy()
        mach_columns = [col for col in df_power_av.columns if col.startswith("Mach_")]
        
        for col in mach_columns:
            df_SEPower[col] = df_power_av[col] - df_power_req[col]
            # Reemplazar valores negativos con NaN
            df_SEPower[col] = df_SEPower[col].where(df_SEPower[col] >= 0, np.nan)
        
        return df_SEPower
    #
    def Max_Specific_excess_power(self, df_SEPower: pd.DataFrame) -> List[Tuple[float, float]]:
        """
        Encuentra el máximo excedente de potencia específica para cada altitud.
        
        Args:
            df_SEPower: DataFrame resultante de Specific_excess_power()
            
        Returns:
            Lista de tuplas (altitud_km, max_SEPower) ordenada por altitud, donde:
            - altitud_km: Altitud en kilómetros
            - max_SEPower: Máximo excedente de potencia específica para esa altitud
        """
        results = []
        mach_columns = [col for col in df_SEPower.columns if col.startswith("Mach_")]
        
        for idx, row in df_SEPower.iterrows():
            alt_km = row['Altitude_km']
            # Encontrar el máximo valor no-NaN para esta altitud
            valid_values = [row[col] for col in mach_columns if not np.isnan(row[col])]
            
            if valid_values:
                max_SEPower = max(valid_values)
                # Encontrar el Mach correspondiente al máximo
                max_mach = next(
                    float(col.split('_')[1]) 
                    for col in mach_columns 
                    if not np.isnan(row[col]) and row[col] == max_SEPower
                )
                results.append((alt_km, max_SEPower, max_mach))
            else:
                # Si todos son NaN para esta altitud
                results.append((alt_km, np.nan, np.nan))
        
        return results
    
    #
    def Max_True_Speed(self, df_SEPower: pd.DataFrame) -> List[Tuple[float, float]]:
        """
        Encuentra la última velocidad verdadera válida (máxima) para cada altitud donde existe excedente de potencia.
        
        Args:
            df_SEPower: DataFrame resultante de Specific_excess_power()
            
        Returns:
            Lista de tuplas (altitud_km, true_speed_ms) donde:
            - altitud_km: Altitud en kilómetros
            - true_speed_ms: Última velocidad verdadera válida (m/s) para esa altitud
            - mach: Número de Mach correspondiente
            Los valores pueden ser NaN si no hay excedente de potencia en esa altitud.
        """
        results = []
        mach_columns = [col for col in df_SEPower.columns if col.startswith("Mach_")]
        
        for idx, row in df_SEPower.iterrows():
            alt_km = row['Altitude_km']
            last_valid = None
            
            # Buscar de derecha a izquierda (mayores Mach primero)
            for col in reversed(mach_columns):
                if not np.isnan(row[col]):
                    mach = float(col.split('_')[1])
                    last_valid = (alt_km, self._true_speed(alt_km, mach), mach)
                    break
            
            results.append(last_valid if last_valid else (alt_km, np.nan, np.nan))
        
        return results
    #
    def Stall_True_Speed(self, df_SEPower: pd.DataFrame) -> List[Tuple[float, float, float]]:
        """
        Encuentra la primera velocidad verdadera válida (mínima) para cada altitud donde existe excedente de potencia.
        
        Args:
            df_SEPower: DataFrame resultante de Specific_excess_power()
            
        Returns:
            Lista de tuplas (altitud_km, true_speed_ms, mach) donde:
            - altitud_km: Altitud en kilómetros
            - true_speed_ms: Primera velocidad verdadera válida (m/s) para esa altitud
            - mach: Número de Mach correspondiente
            Los valores pueden ser NaN si no hay excedente de potencia en esa altitud.
        """
        results = []
        mach_columns = [col for col in df_SEPower.columns if col.startswith("Mach_")]
        
        for idx, row in df_SEPower.iterrows():
            alt_km = row['Altitude_km']
            first_valid = None
            
            # Buscar de izquierda a derecha (menores Mach primero)
            for col in mach_columns:
                if not np.isnan(row[col]):
                    mach = float(col.split('_')[1])
                    first_valid = (alt_km, self._true_speed(alt_km, mach), mach)
                    break
            
            results.append(first_valid if first_valid else (alt_km, np.nan, np.nan))
        
        return results
    #
    def plot_speed_envelope(self,
                           df_clmax: pd.DataFrame,
                           max_speeds: List[Tuple[float, float, float]],
                           save_path: Optional[str] = None,
                           show: bool = True,
                           **kwargs) -> plt.Figure:
        """
        Grafica la envolvente de velocidades con interpolación suavizada para V_max.
        """
        # Configuración de estilo
        plt.style.use(kwargs.get('style', 'seaborn'))
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
        # 1. Procesar datos
        stall_alts = df_clmax['Altitude_km'].values
        stall_vel = df_clmax['V_min_ms'].values
        max_alts = np.array([x[0] for x in max_speeds])
        max_vel = np.array([x[1] for x in max_speeds])
    
        # 2. Interpolación suavizada con splines cúbicos
        def smooth_interp(x_new, x_orig, y_orig):
            from scipy.interpolate import CubicSpline, Akima1DInterpolator
            mask = ~np.isnan(y_orig)
            if sum(mask) < 4:  # Mínimo 4 puntos para spline cúbico
                return interp1d(x_orig[mask], y_orig[mask], 
                              kind='quadratic', 
                              fill_value='extrapolate')(x_new)
            
            # Usar Akima para evitar oscilaciones excesivas
            interp = Akima1DInterpolator(x_orig[mask], y_orig[mask])
            return interp(x_new)
    
        # 3. Generar más puntos para curva suave
        fine_alts = np.linspace(stall_alts.min(), stall_alts.max(), 100)
        max_vel_smooth = smooth_interp(fine_alts, max_alts, max_vel)
    
        # 4. Plot envolvente
        fill_alpha = kwargs.get('fill_alpha', 0.2)
        ax.fill_betweenx(fine_alts, 
                        np.interp(fine_alts, stall_alts, stall_vel),
                        max_vel_smooth,
                        color='limegreen', 
                        alpha=fill_alpha)
    
        # 5. Curvas suavizadas
        ax.plot(stall_vel, stall_alts, 'r-', lw=2.5, label='Límite de stall (V_min)')
        ax.plot(max_vel_smooth, fine_alts, 'b-', lw=2.5, label='Límite aerodinámico (V_max)')
        
        # 6. Puntos originales (opcional)
        # if kwargs.get('show_original', True):
        #     ax.scatter(max_vel, max_alts, c='gold', s=60, edgecolors='navy',
        #               label='Datos originales', zorder=3)
    
        # 7. Configuración del gráfico
        ax.set(xlabel='Velocidad verdadera [m/s]', ylabel='Altitud [km]',
               title=f'{self.aircraft_name} - Envolvente de Velocidades\n(Interpolación Akima)')
        ax.grid(True, ls='--', alpha=0.6)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    #
        