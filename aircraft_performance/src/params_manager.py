# -*- coding: utf-8 -*-
"""
Created on Sat May  3 23:00:24 2025

@author: Rafael Trujillo
"""
from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path
import jsonschema

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aircraft import Aircraft
    
class ParametersManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Carga el schema de validación"""
        schema_path = self.config_dir / "_schemas" / "params_schema.json"
        with open(schema_path, 'r') as f:
            return json.load(f)
    
    def load_parameters(self, aircraft_name: str, params_name: str = "flight_analysis") -> Dict[str, Any]:
        """
        Carga parámetros de análisis para una aeronave específica
        """
        params_path = self.config_dir / "params" / f"{aircraft_name}.json"
        with open(params_path, 'r') as f:
            params = json.load(f)
        # Validación centralizada (aquí se ejecuta al cargar los parámetros)
        if "analysis_name" not in params:
            raise ValueError(
                f"El archivo {params_path} debe incluir 'analysis_name'. "
                f"Ejemplo: 'performance_standard', 'emergency', etc."
            )
        
        if not params_path.exists():
            params_path = self.config_dir / "params" / f"{params_name}.json"
        
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        
        jsonschema.validate(instance=params, schema=self.schema)
        return params
    
    def generate_metadata(self, aircraft: 'Aircraft', params: Dict[str, Any]) -> Dict[str, Any]:
        """Genera metadatos para guardar con los resultados"""
        #from .aircraft import Aircraft  # Importación retardada aquí
        
        return {
            "aircraft_config": {
                "name": aircraft.name,
                "mass_kg": aircraft.mass_kg,
                "wing_area_m2": aircraft.wing_area_m2
            },
            "engine_config": {
                "name": aircraft.engine.name,
                "power_watts": aircraft.engine.power_watts
            },
            "analysis_parameters": params,
            "timestamp": datetime.now().isoformat(),
            "git_hash": self._get_git_hash()
        }
    
    @staticmethod
    def _get_git_hash() -> str:
        """Obtiene el hash de git actual si está disponible"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except:
            return "unknown"