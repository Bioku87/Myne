import sys
import json
import re
import math
import logging
import traceback
from datetime import datetime
from collections import defaultdict
import threading
from functools import lru_cache
import tempfile
import subprocess
from pathlib import Path
import numpy as np

# GUI and visualization imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QFileDialog, QTabWidget, QGroupBox,
                            QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
                            QMessageBox, QProgressBar, QSplitter, QFrame, QDialog,
                            QListWidget, QScrollArea, QFormLayout, QTreeWidget, QTreeWidgetItem,
                            QToolButton, QMenu, QAction, QLineEdit, QRadioButton, QButtonGroup,
                            QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDate, QRect, QSize, QPoint
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QColor, QPainter, QPen, QPolygon

# Try to import optional libraries with fallback mechanisms
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Pandas not available. Excel import/export will be limited.")

try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("XlsxWriter not available. Excel export will be disabled.")

# Try to import CAD libraries with fallback mechanisms
CAD_LIBRARIES = []

# Try ezdxf with ODA File Converter
try:
    import ezdxf
    from ezdxf.addons import odafc
    EZDXF_AVAILABLE = True
    CAD_LIBRARIES.append("ezdxf")
    
    # Check if ODA File Converter is available
    try:
        odafc_path = odafc.get_odafc_path()
        ODA_AVAILABLE = odafc_path is not None
    except Exception:
        ODA_AVAILABLE = False
except ImportError:
    EZDXF_AVAILABLE = False
    ODA_AVAILABLE = False

# Try pyautocad for direct AutoCAD access
try:
    import pyautocad
    PYAUTOCAD_AVAILABLE = True
    CAD_LIBRARIES.append("pyautocad")
except ImportError:
    PYAUTOCAD_AVAILABLE = False

# Try to import OpenCascade if available
try:
    from OCC.Core import TopoDS, BRepTools, BRep
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    OPENCASCADE_AVAILABLE = True
    CAD_LIBRARIES.append("opencascade")
except ImportError:
    OPENCASCADE_AVAILABLE = False

# Set up logging
log_dir = os.path.join(os.path.expanduser("~"), ".solareye")
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
log_filename = os.path.join(log_dir, f"solareye_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SolarEye')
logger.info("Application started")

# Add console handler to see logs in console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Constants
SOLAR_COMPONENT_TYPES = [
    'Block',
    'Inverter',
    'Combiner',
    'String',
    'Module',
    'Tracker'
]

# Regular expression patterns for component identification
DEFAULT_NAME_PATTERNS = {
    'block': r'B[_-]?(\d+)|BLOCK[_-]?(\d+)|ARRAY[_-]?(\d+)|PCU[_-]?(\d+)',
    'inverter': r'INV[_-]?(\d+)|INVERTER[_-]?(\d+)|I[_-]?(\d+)',
    'combiner': r'COMB[_-]?(\d+)|CB[_-]?(\d+)|C[_-]?(\d+)|COMBINER[_-]?BOX[_-]?(\d+)',
    'string': r'STR[_-]?(\d+)|S[_-]?(\d+)|STRING[_-]?(\d+)',
    'module': r'PV[_-]?(\d+)|MODULE[_-]?(\d+)|M[_-]?(\d+)',
    'tracker': r'TRK[_-]?(\d+)|TRACKER[_-]?(\d+)|T[_-]?(\d+)'
}

# Additional pattern sets to improve entity detection from CAD files
ELECTRICAL_DIAGRAM_PATTERNS = {
    'inverter': r'PCS[_-]?(\d+)|CENTRAL[_-]?INVERTER[_-]?(\d+)|STATION[_-]?(\d+)',
    'combiner': r'DCDB[_-]?(\d+)|SMB[_-]?(\d+)|STRING[_-]?MONITOR[_-]?(\d+)',
    'string': r'PANEL[_-]?STRING[_-]?(\d+)|PV[_-]?STRING[_-]?(\d+)',
    'transformer': r'XFMR[_-]?(\d+)|TRANSFORMER[_-]?(\d+)'
}

LAYER_NAME_HINTS = {
    'block': ['block', 'array', 'pcu', 'section'],
    'inverter': ['inverter', 'inv', 'power', 'station'],
    'combiner': ['combiner', 'dcdb', 'smb', 'junction', 'jb'],
    'string': ['string', 'str', 'circuit'],
    'module': ['module', 'panel', 'pv'],
    'tracker': ['tracker', 'tracking', 'mount']
}

# Solar-specific naming conventions often found in CAD files
SOLAR_NAMING_CONVENTIONS = {
    'module': ['panel', 'pv module', 'solar module', 'panel array'],
    'tracker': ['tracking system', 'single axis', 'dual axis', 'horizontal tracker'],
    'inverter': ['power station', 'inverter station', 'pcs', 'power block'],
    'combiner': ['dc combiner', 'string combiner', 'harness', 'junction box']
}

class MemoryManaged:
    """Mixin class for memory management techniques."""
    
    @staticmethod
    def process_in_chunks(data, chunk_size, process_func):
        """Process data in chunks to reduce memory usage."""
        if not data:
            return []
        
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            results.extend(process_func(chunk))
        
        return results
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_process(data_id, **kwargs):
        """Cache results of expensive operations."""
        # This is a placeholder for actual processing logic
        # The real implementation would depend on specific needs
        pass
    
    @staticmethod
    def process_by_region(entities, bounds, region_size, process_func):
        """Process entities by spatial region."""
        if not entities:
            return []
            
        min_x, min_y, max_x, max_y = bounds
        
        # Calculate regions
        x_regions = math.ceil((max_x - min_x) / region_size[0])
        y_regions = math.ceil((max_y - min_y) / region_size[1])
        
        results = []
        
        # Group entities by region
        region_entities = defaultdict(list)
        
        for entity in entities:
            entity_pos = MemoryManaged.get_entity_position(entity)
            if not entity_pos:
                continue
                
            x, y = entity_pos
            
            # Determine region indices
            x_idx = min(max(int((x - min_x) / region_size[0]), 0), x_regions - 1)
            y_idx = min(max(int((y - min_y) / region_size[1]), 0), y_regions - 1)
            
            region_key = (x_idx, y_idx)
            region_entities[region_key].append(entity)
        
        # Process each region
        for region_key, entities in region_entities.items():
            region_results = process_func(entities)
            results.extend(region_results)
        
        return results
    
    @staticmethod
    def get_entity_position(entity):
        """Extract position from an entity."""
        # This is a generic implementation - specific adapters would override this
        if hasattr(entity, 'position'):
            return entity.position
        elif hasattr(entity, 'insertion_point'):
            return entity.insertion_point
        elif hasattr(entity, 'center'):
            return entity.center
        elif hasattr(entity, 'start'):
            # For line entities, use midpoint
            if hasattr(entity, 'end'):
                return ((entity.start[0] + entity.end[0]) / 2, 
                        (entity.start[1] + entity.end[1]) / 2)
            return entity.start
        return None

class CADPort:
    """Port interface for CAD file operations."""
    
    def read_file(self, file_path):
        """Read a CAD file and return its data."""
        raise NotImplementedError("Subclasses must implement read_file")
    
    def extract_entities(self, cad_data):
        """Extract entities from CAD data."""
        raise NotImplementedError("Subclasses must implement extract_entities")
    
    def extract_blocks(self, cad_data):
        """Extract blocks from CAD data."""
        raise NotImplementedError("Subclasses must implement extract_blocks")
    
    def identify_component(self, entity, surrounding_entities=None):
        """Identify what solar component an entity represents."""
        raise NotImplementedError("Subclasses must implement identify_component")

class SiteStructurePort:
    """Port interface for site structure operations."""
    
    def create_structure(self):
        """Create a new site structure."""
        raise NotImplementedError("Subclasses must implement create_structure")
    
    def add_block(self, block_id, params=None):
        """Add a block to the structure."""
        raise NotImplementedError("Subclasses must implement add_block")
    
    def add_inverter(self, block_id, inverter_id, params=None):
        """Add an inverter to a block."""
        raise NotImplementedError("Subclasses must implement add_inverter")
    
    def add_combiner(self, block_id, inverter_id, combiner_id, params=None):
        """Add a combiner to an inverter."""
        raise NotImplementedError("Subclasses must implement add_combiner")
    
    def add_string(self, block_id, inverter_id, combiner_id, string_id, params=None):
        """Add a string to a combiner."""
        raise NotImplementedError("Subclasses must implement add_string")
    
    def add_module(self, block_id, inverter_id, combiner_id, string_id, module_id, params=None):
        """Add a module to a string."""
        raise NotImplementedError("Subclasses must implement add_module")
    
    def add_tracker(self, block_id, tracker_id, params=None):
        """Add a tracker to a block."""
        raise NotImplementedError("Subclasses must implement add_tracker")
    
    def export_structure(self, format_type, file_path=None):
        """Export the structure to the specified format."""
        raise NotImplementedError("Subclasses must implement export_structure")


class DWGConverterUtil:
    """Utility class for handling DWG file conversion."""
    
    @staticmethod
    def find_conversion_tools():
        """Find available tools for DWG conversion."""
        conversion_tools = []
        
        # Check for ODA File Converter
        if ODA_AVAILABLE:
            conversion_tools.append(("ODA File Converter", "oda"))
        
        # Check for AutoCAD (via system paths)
        acad_paths = []
        
        # Windows
        if sys.platform == 'win32':
            program_files = [
                os.environ.get('ProgramFiles', r'C:\Program Files'),
                os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
            ]
            
            for pf in program_files:
                # Check for various AutoCAD versions
                for year in range(2015, 2025):
                    acad_path = os.path.join(pf, f"Autodesk\\AutoCAD {year}\\acad.exe")
                    if os.path.exists(acad_path):
                        acad_paths.append((f"AutoCAD {year}", acad_path))
        
        # MacOS
        elif sys.platform == 'darwin':
            # Check for AutoCAD for Mac
            for year in range(2015, 2025):
                acad_path = f"/Applications/Autodesk/AutoCAD {year}/AutoCAD {year}.app"
                if os.path.exists(acad_path):
                    acad_paths.append((f"AutoCAD {year}", acad_path))
        
        # Linux (limited support)
        elif sys.platform.startswith('linux'):
            # Check Wine paths
            wine_prefix = os.environ.get('WINEPREFIX', os.path.expanduser('~/.wine'))
            program_files = os.path.join(wine_prefix, "drive_c", "Program Files")
            program_files_x86 = os.path.join(wine_prefix, "drive_c", "Program Files (x86)")
            
            for pf in [program_files, program_files_x86]:
                for year in range(2015, 2025):
                    acad_path = os.path.join(pf, f"Autodesk\\AutoCAD {year}\\acad.exe")
                    if os.path.exists(acad_path):
                        acad_paths.append((f"AutoCAD {year} (Wine)", acad_path))
        
        # Add AutoCAD if found
        if acad_paths:
            # Use the most recent version
            newest_acad = acad_paths[-1]
            conversion_tools.append((newest_acad[0], "autocad"))
        
        # Check for LibreCAD (for DXF at least)
        librecad_path = None
        if sys.platform == 'win32':
            # Windows - check Program Files
            for pf in [r'C:\Program Files', r'C:\Program Files (x86)']:
                lc_path = os.path.join(pf, 'LibreCAD', 'librecad.exe')
                if os.path.exists(lc_path):
                    librecad_path = lc_path
                    break
        elif sys.platform == 'darwin':
            # MacOS
            lc_path = '/Applications/LibreCAD.app/Contents/MacOS/LibreCAD'
            if os.path.exists(lc_path):
                librecad_path = lc_path
        else:
            # Linux - check common paths
            for path in ['/usr/bin/librecad', '/usr/local/bin/librecad']:
                if os.path.exists(path):
                    librecad_path = path
                    break
        
        if librecad_path:
            conversion_tools.append(("LibreCAD", "librecad"))
        
        # Add manual DWG to DXF conversion option
        conversion_tools.append(("Manual DWG to DXF conversion", "manual"))
        
        return conversion_tools
    
    @staticmethod
    def convert_dwg_to_dxf(dwg_path, method="oda", config=None):
        """Convert DWG file to DXF format.
        
        Args:
            dwg_path: Path to the DWG file
            method: Conversion method ('oda', 'autocad', 'librecad', or 'manual')
            config: Additional configuration for the selected method
        
        Returns:
            Path to the generated DXF file or None if conversion failed
        """
        if not os.path.exists(dwg_path):
            logger.error(f"DWG file not found: {dwg_path}")
            return None
        
        base_dir = os.path.dirname(dwg_path)
        base_name = os.path.splitext(os.path.basename(dwg_path))[0]
        dxf_path = os.path.join(base_dir, base_name + ".dxf")
        
        # ODA File Converter method
        if method == "oda" and ODA_AVAILABLE:
            try:
                logger.info(f"Converting DWG to DXF using ODA File Converter: {dwg_path}")
                dxf_path = odafc.convert(dwg_path, version="R2018", audit=True)
                if os.path.exists(dxf_path):
                    logger.info(f"Conversion successful: {dxf_path}")
                    return dxf_path
                else:
                    logger.error("Conversion failed - DXF file not created")
                    return None
            except Exception as e:
                logger.error(f"Error during ODA conversion: {str(e)}")
                return None
        
        # AutoCAD method (through scripting)
        elif method == "autocad":
            try:
                acad_path = config.get('acad_path')
                if not acad_path or not os.path.exists(acad_path):
                    logger.error(f"AutoCAD executable not found: {acad_path}")
                    return None
                
                # Create temporary script file for AutoCAD
                script_path = os.path.join(tempfile.gettempdir(), "dwgconvert.scr")
                with open(script_path, 'w') as f:
                    # Escape backslashes in paths for AutoCAD script
                    safe_dwg_path = dwg_path.replace('\\', '/')
                    safe_dxf_path = dxf_path.replace('\\', '/')
                    
                    # Script: open DWG, save as DXF, quit
                    f.write(f'OPEN "{safe_dwg_path}"\n')
                    f.write(f'SAVEAS DXF 2018 "{safe_dxf_path}"\n')
                    f.write('QUIT Y\n')
                
                # Run AutoCAD with script
                if sys.platform == 'win32':
                    subprocess.call([acad_path, "/b", script_path])
                elif sys.platform == 'darwin':
                    # MacOS specific command
                    subprocess.call(["open", "-a", acad_path, "--args", "/b", script_path])
                else:
                    # Linux with Wine
                    subprocess.call(["wine", acad_path, "/b", script_path])
                
                # Check if DXF file was created
                if os.path.exists(dxf_path):
                    logger.info(f"AutoCAD conversion successful: {dxf_path}")
                    return dxf_path
                else:
                    logger.error("AutoCAD conversion failed - DXF file not created")
                    return None
                
            except Exception as e:
                logger.error(f"Error during AutoCAD conversion: {str(e)}")
                return None
        
        # LibreCAD method (limited support for DWG)
        elif method == "librecad":
            try:
                librecad_path = config.get('librecad_path')
                if not librecad_path or not os.path.exists(librecad_path):
                    logger.error(f"LibreCAD executable not found: {librecad_path}")
                    return None
                
                # Run LibreCAD
                subprocess.call([librecad_path, "-o", dxf_path, dwg_path])
                
                # Check if DXF file was created
                if os.path.exists(dxf_path):
                    logger.info(f"LibreCAD conversion successful: {dxf_path}")
                    return dxf_path
                else:
                    logger.error("LibreCAD conversion failed - DXF file not created")
                    return None
                
            except Exception as e:
                logger.error(f"Error during LibreCAD conversion: {str(e)}")
                return None
        
        # Manual method (prompt user to convert)
        elif method == "manual":
            logger.info("Manual DWG to DXF conversion requested")
            # Just return the expected DXF path - the calling code will check if it exists
            return dxf_path
        
        else:
            logger.error(f"Unsupported conversion method: {method}")
            return None

class DWGConversionDialog(QDialog):
    """Dialog for DWG to DXF conversion options."""
    
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.conversion_method = None
        self.conversion_config = {}
        
        self.setWindowTitle("DWG to DXF Conversion")
        self.setMinimumWidth(500)
        
        self._create_ui()
    
    def _create_ui(self):
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(
            f"The file <b>{os.path.basename(self.file_path)}</b> needs to be "
            f"converted to DXF format for processing. Please select a conversion method:"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Find available conversion tools
        self.conversion_tools = DWGConverterUtil.find_conversion_tools()
        
        # Create radio buttons for conversion methods
        self.method_group = QButtonGroup(self)
        
        for i, (tool_name, tool_id) in enumerate(self.conversion_tools):
            radio = QRadioButton(tool_name)
            self.method_group.addButton(radio, i)
            layout.addWidget(radio)
            
            # Select first option by default
            if i == 0:
                radio.setChecked(True)
        
        # Add note for manual conversion
        note_label = QLabel(
            "Note: If you choose 'Manual DWG to DXF conversion', you'll need to convert "
            "the file yourself using AutoCAD or another CAD program, and save it with "
            "the same name but .dxf extension."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666; font-style: italic")
        layout.addWidget(note_label)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        convert_button = QPushButton("Convert")
        convert_button.clicked.connect(self.accept)
        button_layout.addWidget(convert_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def get_selected_method(self):
        """Get the selected conversion method and config."""
        selected_id = self.method_group.checkedId()
        
        if selected_id >= 0 and selected_id < len(self.conversion_tools):
            _, method_id = self.conversion_tools[selected_id]
            return method_id, self.conversion_config
        
        return None, {}

class EnhancedDWGEzdxfAdapter(CADPort, MemoryManaged):
    """Enhanced adapter for ezdxf library with improved DWG support."""
    
    def __init__(self, name_patterns=None, additional_patterns=None, layer_hints=None, parent_widget=None):
        self.name_patterns = name_patterns or DEFAULT_NAME_PATTERNS
        self.additional_patterns = additional_patterns or ELECTRICAL_DIAGRAM_PATTERNS
        self.layer_hints = layer_hints or LAYER_NAME_HINTS
        self.parent_widget = parent_widget  # For dialog display
        
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf library not available")
        
        # Store detected layers for faster component inference
        self.layer_types = {}
        # Store blocks for context
        self.blocks = {}
        # Store bounds for spatial analysis
        self.bounds = None
        # Store all entities by handle for cross-referencing
        self.entities_by_handle = {}
        # Keep track of parent-child relationships
        self.parent_child_map = defaultdict(list)
        # Store detected components
        self.detected_components = []
    
    def read_file(self, file_path):
        """Read a CAD file using ezdxf with enhanced DWG support."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            doc = None
            
            if ext == '.dwg':
                # Check for ODA availability first
                if ODA_AVAILABLE:
                    logger.info(f"Converting DWG to DXF using ODA File Converter: {file_path}")
                    try:
                        dxf_file = odafc.convert(file_path)
                        doc = ezdxf.readfile(dxf_file)
                        logger.info("ODA conversion successful")
                    except Exception as e:
                        logger.error(f"ODA conversion failed: {str(e)}")
                        doc = None
                
                # If ODA failed or not available, try alternative methods
                if doc is None:
                    try:
                        # Show the conversion dialog
                        if self.parent_widget:
                            dialog = DWGConversionDialog(file_path, self.parent_widget)
                            result = dialog.exec_()
                            
                            if result == QDialog.Accepted:
                                method, config = dialog.get_selected_method()
                                dxf_file = DWGConverterUtil.convert_dwg_to_dxf(file_path, method, config)
                                
                                if dxf_file and os.path.exists(dxf_file):
                                    doc = ezdxf.readfile(dxf_file)
                                    logger.info(f"DWG conversion successful using {method}")
                                else:
                                    raise ValueError("Conversion failed - no DXF file created")
                            else:
                                raise ValueError("DWG conversion cancelled by user")
                        else:
                            # No parent widget, try available methods without dialog
                            for method, _ in DWGConverterUtil.find_conversion_tools():
                                if method[1] != "manual":  # Skip manual method without dialog
                                    dxf_file = DWGConverterUtil.convert_dwg_to_dxf(file_path, method[1])
                                    if dxf_file and os.path.exists(dxf_file):
                                        doc = ezdxf.readfile(dxf_file)
                                        break
                            
                            if doc is None:
                                raise ValueError("No automatic conversion method succeeded")
                    except Exception as e:
                        logger.error(f"Alternative DWG conversion failed: {str(e)}")
                        raise ValueError(f"DWG file cannot be processed: {str(e)}")
            
            elif ext == '.dxf':
                # Directly read DXF file
                logger.info(f"Reading DXF file: {file_path}")
                doc = ezdxf.readfile(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Calculate bounds from modelspace entities
            self._calculate_bounds(doc.modelspace())
            
            # Detect layer types for faster component inference
            self._analyze_layers(doc)
            
            # Extract blocks for context
            self.blocks = {block.name: block for block in doc.blocks}
            
            return doc
            
        except Exception as e:
            logger.error(f"ezdxf read error: {str(e)}")
            
            # Try to recover
            try:
                logger.info("Attempting DXF recovery...")
                doc = ezdxf.recover.readfile(file_path)
                logger.info("Recovery successful")
                
                # Calculate bounds from modelspace entities (after recovery)
                self._calculate_bounds(doc.modelspace())
                
                # Detect layer types for faster component inference
                self._analyze_layers(doc)
                
                # Extract blocks for context
                self.blocks = {block.name: block for block in doc.blocks}
                
                return doc
            except Exception as recover_err:
                logger.error(f"Recovery failed: {str(recover_err)}")
                raise
    
    def _calculate_bounds(self, modelspace):
        """Calculate bounds of entities in modelspace."""
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for entity in modelspace:
            if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'handle'):
                pos = self._get_entity_position(entity)
                if pos:
                    x, y = pos
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        
        # If no entities with position found, use defaults
        if min_x == float('inf'):
            min_x, min_y = 0, 0
        if max_x == float('-inf'):
            max_x, max_y = 1000, 1000
        
        self.bounds = (min_x, min_y, max_x, max_y)
        logger.info(f"Calculated bounds: {self.bounds}")
    
    def _analyze_layers(self, doc):
        """Analyze layers to infer their purpose from names."""
        for layer_name in doc.layers:
            layer_name_lower = layer_name.dxf.name.lower()
            
            for component_type, hints in self.layer_hints.items():
                for hint in hints:
                    if hint in layer_name_lower:
                        self.layer_types[layer_name.dxf.name] = component_type
                        break
        
        logger.info(f"Detected layer types: {len(self.layer_types)} layers classified")
    
    def _get_entity_position(self, entity):
        """Extract position from an entity."""
        try:
            if entity.dxftype() == 'TEXT':
                return (entity.dxf.insert.x, entity.dxf.insert.y)
            elif entity.dxftype() == 'MTEXT':
                return (entity.dxf.insert.x, entity.dxf.insert.y)
            elif entity.dxftype() == 'INSERT':
                return (entity.dxf.insert.x, entity.dxf.insert.y)
            elif entity.dxftype() == 'LINE':
                # Use midpoint
                return (
                    (entity.dxf.start.x + entity.dxf.end.x) / 2,
                    (entity.dxf.start.y + entity.dxf.end.y) / 2
                )
            elif entity.dxftype() == 'CIRCLE':
                return (entity.dxf.center.x, entity.dxf.center.y)
            elif entity.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
                # Use center of vertices
                if hasattr(entity, 'vertices') and len(entity.vertices) > 0:
                    vertices = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
                    if vertices:
                        x_sum = sum(v[0] for v in vertices)
                        y_sum = sum(v[1] for v in vertices)
                        return (x_sum / len(vertices), y_sum / len(vertices))
                return None
            else:
                return None
        except Exception as e:
            logger.debug(f"Error getting entity position for {entity.dxftype()}: {str(e)}")
            return None
    
    def extract_entities(self, doc):
        """Extract entities from DXF document."""
        entities = []
        
        try:
            # Access modelspace
            msp = doc.modelspace()
            
            # Extract entities from modelspace
            for entity in msp:
                py_entity = self._convert_entity(entity)
                if py_entity:
                    entities.append(py_entity)
                    # Store by handle for cross-referencing
                    if 'handle' in py_entity:
                        self.entities_by_handle[py_entity['handle']] = py_entity
            
            # Build parent-child relationships based on spatial hierarchy
            self._build_hierarchy(entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"ezdxf entity extraction error: {str(e)}")
            raise
    
    def _convert_entity(self, entity):
        """Convert ezdxf entity to Python dict."""
        try:
            entity_data = {
                'type': entity.dxftype(),
                'handle': entity.dxf.handle,
                'layer': entity.dxf.layer
            }
            
            # Extract extended entity data if available
            if hasattr(entity, 'get_xdata'):
                try:
                    xdata = entity.get_xdata()
                    if xdata:
                        entity_data['xdata'] = xdata
                except:
                    pass
            
            # Extract text entities
            if entity.dxftype() == 'TEXT':
                entity_data['text'] = entity.dxf.text
                entity_data['position'] = (entity.dxf.insert.x, entity.dxf.insert.y)
                entity_data['height'] = entity.dxf.height
                entity_data['rotation'] = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0
                
                # Extract style
                if hasattr(entity.dxf, 'style'):
                    entity_data['style'] = entity.dxf.style
            
            elif entity.dxftype() == 'MTEXT':
                entity_data['text'] = entity.text
                entity_data['position'] = (entity.dxf.insert.x, entity.dxf.insert.y)
                entity_data['height'] = entity.dxf.char_height
                entity_data['rotation'] = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0
                
                # Extract style
                if hasattr(entity.dxf, 'style'):
                    entity_data['style'] = entity.dxf.style
            
            # Extract block reference entities
            elif entity.dxftype() == 'INSERT':
                entity_data['name'] = entity.dxf.name
                entity_data['position'] = (entity.dxf.insert.x, entity.dxf.insert.y)
                entity_data['rotation'] = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0
                entity_data['scale_x'] = entity.dxf.xscale if hasattr(entity.dxf, 'xscale') else 1
                entity_data['scale_y'] = entity.dxf.yscale if hasattr(entity.dxf, 'yscale') else 1
                
                # Extract attributes if available
                if entity.has_attribs:
                    entity_data['attributes'] = {}
                    for attrib in entity.attribs:
                        entity_data['attributes'][attrib.dxf.tag] = attrib.dxf.text
                
                # Check block name against solar components
                block_name = entity.dxf.name.lower()
                for comp_type, terms in SOLAR_NAMING_CONVENTIONS.items():
                    for term in terms:
                        if term in block_name:
                            entity_data['component_hint'] = comp_type
                            break
                    if 'component_hint' in entity_data:
                        break
            
            # Extract line entities
            elif entity.dxftype() == 'LINE':
                entity_data['start'] = (entity.dxf.start.x, entity.dxf.start.y)
                entity_data['end'] = (entity.dxf.end.x, entity.dxf.end.y)
                # Set position as midpoint
                entity_data['position'] = (
                    (entity.dxf.start.x + entity.dxf.end.x) / 2,
                    (entity.dxf.start.y + entity.dxf.end.y) / 2
                )
                entity_data['length'] = math.sqrt(
                    (entity.dxf.end.x - entity.dxf.start.x) ** 2 + 
                    (entity.dxf.end.y - entity.dxf.start.y) ** 2
                )
            
            # Extract circle entities
            elif entity.dxftype() == 'CIRCLE':
                entity_data['center'] = (entity.dxf.center.x, entity.dxf.center.y)
                entity_data['radius'] = entity.dxf.radius
                entity_data['position'] = (entity.dxf.center.x, entity.dxf.center.y)
                entity_data['area'] = math.pi * (entity.dxf.radius ** 2)
            
            # Extract polyline entities
            elif entity.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
                try:
                    # Different approaches for different polyline types
                    if entity.dxftype() == 'LWPOLYLINE':
                        vertices = [(point[0], point[1]) for point in entity.get_points()]
                    else:  # Regular POLYLINE
                        vertices = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
                        
                    entity_data['vertices'] = vertices
                    
                    # Set position as center of vertices
                    if vertices:
                        x_sum = sum(v[0] for v in vertices)
                        y_sum = sum(v[1] for v in vertices)
                        entity_data['position'] = (x_sum / len(vertices), y_sum / len(vertices))
                        
                        # Calculate bounding box and area
                        min_x = min(v[0] for v in vertices)
                        min_y = min(v[1] for v in vertices)
                        max_x = max(v[0] for v in vertices)
                        max_y = max(v[1] for v in vertices)
                        
                        entity_data['bbox'] = (min_x, min_y, max_x, max_y)
                        entity_data['width'] = max_x - min_x
                        entity_data['height'] = max_y - min_y
                        entity_data['area'] = entity_data['width'] * entity_data['height']
                        
                        # Check if polyline is closed (might be a solar panel or inverter outline)
                        entity_data['is_closed'] = entity.is_closed
                        
                        # Simple shape detection for rectangles (common for solar components)
                        if len(vertices) == 4 or len(vertices) == 5:  # 5th vertex same as 1st in closed polyline
                            # Check if it's a rectangle (perpendicular sides)
                            sides = []
                            for i in range(len(vertices) - 1):
                                v1 = vertices[i]
                                v2 = vertices[i + 1]
                                sides.append((v2[0] - v1[0], v2[1] - v1[1]))
                            
                            # Add last side connecting back to first vertex
                            if len(vertices) == 4:  # Not closed yet
                                v1 = vertices[-1]
                                v2 = vertices[0]
                                sides.append((v2[0] - v1[0], v2[1] - v1[1]))
                            
                            # Check for perpendicular sides
                            is_rectangle = True
                            for i in range(len(sides)):
                                s1 = sides[i]
                                s2 = sides[(i + 1) % len(sides)]
                                # Dot product should be close to 0 for perpendicular sides
                                dot_product = s1[0] * s2[0] + s1[1] * s2[1]
                                if abs(dot_product) > 0.1:  # Allow for slight imprecision
                                    is_rectangle = False
                                    break
                            
                            entity_data['is_rectangle'] = is_rectangle
                            
                            # Rectangles are often solar modules or inverters
                            if is_rectangle:
                                # Classify by size
                                area = entity_data['area']
                                if 1 <= area <= 10:  # Small rectangle, likely a module
                                    entity_data['component_hint'] = 'module'
                                elif 10 < area <= 100:  # Medium rectangle, could be combiner
                                    entity_data['component_hint'] = 'combiner'
                                elif area > 100:  # Large rectangle, might be inverter or block
                                    entity_data['component_hint'] = 'inverter'
                except Exception as e:
                    logger.debug(f"Error processing polyline: {str(e)}")
            
            # Extract hatch entities (often used for solar modules)
            elif entity.dxftype() == 'HATCH':
                try:
                    entity_data['pattern'] = entity.dxf.pattern_name
                    
                    # Get bounding box
                    all_vertices = []
                    for path in entity.paths:
                        for vertex in path.vertices:
                            if hasattr(vertex, 'x') and hasattr(vertex, 'y'):
                                all_vertices.append((vertex.x, vertex.y))
                    
                    if all_vertices:
                        min_x = min(v[0] for v in all_vertices)
                        min_y = min(v[1] for v in all_vertices)
                        max_x = max(v[0] for v in all_vertices)
                        max_y = max(v[1] for v in all_vertices)
                        
                        entity_data['bbox'] = (min_x, min_y, max_x, max_y)
                        entity_data['width'] = max_x - min_x
                        entity_data['height'] = max_y - min_y
                        entity_data['area'] = entity_data['width'] * entity_data['height']
                        entity_data['position'] = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                        
                        # Solar modules often appear as hatches
                        if 1 <= entity_data['area'] <= 10:
                            entity_data['component_hint'] = 'module'
                except Exception as e:
                    logger.debug(f"Error processing hatch: {str(e)}")
            
            # Add layer information and check for solar-specific layers
            if entity_data['layer'] in self.layer_types:
                entity_data['layer_type'] = self.layer_types[entity_data['layer']]
            
            layer_name = entity.dxf.layer.lower()
            for comp_type, hints in self.layer_hints.items():
                for hint in hints:
                    if hint in layer_name:
                        entity_data['layer_type'] = comp_type
                        break
                if 'layer_type' in entity_data:
                    break
            
            # Check for color hints (some CAD systems use color coding for component types)
            if hasattr(entity.dxf, 'color'):
                entity_data['color'] = entity.dxf.color
            
            return entity_data
        
        except Exception as e:
            logger.debug(f"Error converting entity: {str(e)}")
            return None
    
    def _build_hierarchy(self, entities):
        """Build parent-child relationships based on spatial hierarchy."""
        # Sort entities by area (if available) - larger entities first
        sized_entities = []
        for entity in entities:
            area = None
            if 'area' in entity:
                area = entity['area']
            elif 'width' in entity and 'height' in entity:
                area = entity['width'] * entity['height']
            elif 'radius' in entity:
                area = math.pi * entity['radius'] ** 2
            
            if area is not None:
                sized_entities.append((area, entity))
        
        # Sort by area, largest first
        sized_entities.sort(reverse=True)
        
        # Establish parent-child relationships based on containment
        for i, (area1, entity1) in enumerate(sized_entities):
            if 'position' not in entity1 or ('bbox' not in entity1 and 'radius' not in entity1):
                continue
            
            for j, (area2, entity2) in enumerate(sized_entities[i+1:]):
                if 'position' not in entity2:
                    continue
                
                # Skip if entities are of the same type (avoid nesting similar components)
                if ('component_hint' in entity1 and 'component_hint' in entity2 and 
                    entity1['component_hint'] == entity2['component_hint']):
                    continue
                
                # Check if entity2 is contained within entity1
                contained = False
                
                if 'bbox' in entity1:
                    min_x, min_y, max_x, max_y = entity1['bbox']
                    x2, y2 = entity2['position']
                    if min_x <= x2 <= max_x and min_y <= y2 <= max_y:
                        contained = True
                elif 'radius' in entity1:
                    x1, y1 = entity1['position']
                    x2, y2 = entity2['position']
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance <= entity1['radius']:
                        contained = True
                
                if contained:
                    self.parent_child_map[entity1['handle']].append(entity2['handle'])
        
        # Additional check for entity names that indicate parent-child relationships
        for entity in entities:
            if 'name' not in entity:
                continue
            
            name = entity['name'].lower()
            
            # Look for block references with names implying parent-child
            for other in entities:
                if 'name' not in other or other['handle'] == entity['handle']:
                    continue
                
                other_name = other['name'].lower()
                
                # Check for parent-child naming convention
                # e.g. "Inverter01" and "InvBox01" or "Inv01_Combiner02"
                for component_base in ['inv', 'inverter', 'cb', 'combiner', 'string', 'tracker']:
                    if component_base in name and component_base in other_name:
                        # Extract numeric identifiers
                        name_numbers = re.findall(r'\d+', name)
                        other_numbers = re.findall(r'\d+', other_name)
                        
                        if name_numbers and other_numbers:
                            # If first number matches, might be parent-child
                            if name_numbers[0] == other_numbers[0]:
                                # Longer name (with more numbers) is likely child
                                if len(other_numbers) > len(name_numbers):
                                    self.parent_child_map[entity['handle']].append(other['handle'])
                                elif len(name_numbers) > len(other_numbers):
                                    self.parent_child_map[other['handle']].append(entity['handle'])
    
    def extract_blocks(self, doc):
        """Extract block definitions from DXF document."""
        blocks = []
        
        try:
            # Extract from block entities
            for block in doc.blocks:
                block_data = {
                    'name': block.name,
                    'entities': []
                }
                
                # Check if block name suggests a specific component type
                block_name = block.name.lower()
                for comp_type, terms in SOLAR_NAMING_CONVENTIONS.items():
                    for term in terms:
                        if term in block_name:
                            block_data['component_type'] = comp_type
                            break
                    if 'component_type' in block_data:
                        break
                
                # For standard solar module blocks with common names
                for module_pattern in ['module', 'panel', 'pv', 'array']:
                    if module_pattern in block_name:
                        block_data['component_type'] = 'module'
                        break
                
                # Extract entities in block
                for entity in block:
                    py_entity = self._convert_entity(entity)
                    if py_entity:
                        block_data['entities'].append(py_entity)
                
                blocks.append(block_data)
            
            return blocks
            
        except Exception as e:
            logger.error(f"ezdxf block extraction error: {str(e)}")
            raise
    
    def _extract_id_number(self, match_result):
        """Extract ID number from regex match result."""
        if not match_result:
            return None
        
        # Find first non-None group
        for group in match_result.groups():
            if group is not None:
                return group
        
        return None
    
    def identify_component(self, entity, surrounding_entities=None):
        """Identify what solar component an entity represents."""
        # Extract text for identification
        text = None
        component_info = {}
        
        # 1. Check for direct component hints from entity conversion
        if 'component_hint' in entity:
            component_info['component_hint'] = entity['component_hint']
        
        # 2. Check layer type - it's a strong clue
        if 'layer_type' in entity:
            component_info['layer_type'] = entity['layer_type']
        
        # 3. Extract text or attribute values for identification
        if entity.get('type') in ('TEXT', 'MTEXT'):
            text = entity.get('text', '')
            component_info['text'] = text
        elif entity.get('type') == 'INSERT':
            # Check attributes for identification text
            if 'attributes' in entity:
                for tag, value in entity['attributes'].items():
                    if tag.upper() in ('ID', 'NAME', 'TAG', 'LABEL', 'TEXT'):
                        text = value
                        component_info['text'] = text
                        break
                
                if text is None and len(entity['attributes']) > 0:
                    # Use first attribute value if no specific tag found
                    text = next(iter(entity['attributes'].values()))
                    component_info['text'] = text
            
            # Use block name as fallback
            if text is None and 'name' in entity:
                text = entity['name']
                component_info['block_name'] = text
        
        if not text and not 'component_hint' in entity:
            # Try to infer from context without text
            return self._infer_from_context(entity, surrounding_entities)
        
        # 4. Determine component type and ID
        component_type = None
        component_id = None
        
        # Use component_hint if available
        if 'component_hint' in entity:
            hint = entity['component_hint']
            component_type = hint.capitalize()
            
            # Try to extract ID from text or generate from position
            if text:
                id_match = re.search(r'(\d+)', text)
                if id_match:
                    component_id = id_match.group(1)
                else:
                    # Generate ID from position
                    if 'position' in entity:
                        x, y = entity['position']
                        component_id = f"{int(x/10)}-{int(y/10)}"
            elif 'position' in entity:
                # Generate ID from position
                x, y = entity['position']
                component_id = f"{int(x/10)}-{int(y/10)}"
        
        # If not determined by hint, check text patterns
        if component_type is None and text:
            # Check inverter pattern first (most specific)
            inverter_match = re.search(self.name_patterns['inverter'], text, re.IGNORECASE)
            if inverter_match:
                component_type = 'Inverter'
                component_id = self._extract_id_number(inverter_match)
            
            # Check additional electrical diagram patterns for inverters
            if component_type is None and 'inverter' in self.additional_patterns:
                add_inverter_match = re.search(self.additional_patterns['inverter'], text, re.IGNORECASE)
                if add_inverter_match:
                    component_type = 'Inverter'
                    component_id = self._extract_id_number(add_inverter_match)
            
            # If not an inverter, check other patterns
            if component_type is None:
                for pattern_type, pattern in self.name_patterns.items():
                    if pattern_type == 'inverter':
                        continue  # Already checked
                    
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        component_type = pattern_type.capitalize()
                        component_id = self._extract_id_number(match)
                        break
                
                # If still no match, check additional patterns
                if component_type is None:
                    for pattern_type, pattern in self.additional_patterns.items():
                        if pattern_type == 'inverter':
                            continue  # Already checked
                        
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            component_type = pattern_type.capitalize()
                            component_id = self._extract_id_number(match)
                            break
        
        # 5. If still no match but has numbers, try to infer type from context
        if component_type is None and text and re.search(r'\d+', text):
            # Try to infer from layer type
            if 'layer_type' in entity:
                component_type = entity['layer_type'].capitalize()
                # Try to extract ID from text
                id_match = re.search(r'(\d+)', text)
                if id_match:
                    component_id = id_match.group(1)
            else:
                # Check layer name for clues
                layer = entity.get('layer', '').lower()
                for comp_type, hints in self.layer_hints.items():
                    for hint in hints:
                        if hint in layer:
                            component_type = comp_type.capitalize()
                            # Try to extract ID from text
                            id_match = re.search(r'(\d+)', text)
                            if id_match:
                                component_id = id_match.group(1)
                            break
                    if component_type:
                        break
        
        # 6. If component type and ID found, create component object
        if component_type and component_id:
            component = {
                'type': component_type,
                'id': component_id,
                'text': text,
                'handle': entity.get('handle'),
                'position': entity.get('position'),
                'source_entity': entity
            }
            
            # If we've already detected this component in a previous pass, skip it
            # This prevents duplicates when multiple entities refer to the same component
            if self._is_duplicate_component(component):
                return None
            
            # Add to detected components list
            self.detected_components.append(component)
            return component
        
        return None
    
    def _is_duplicate_component(self, component):
        """Check if a component with the same type and ID has already been detected."""
        for existing in self.detected_components:
            if (existing['type'] == component['type'] and 
                existing['id'] == component['id']):
                return True
        return False
    
    def _infer_from_context(self, entity, surrounding_entities=None):
        """Infer component type from context when no text is available."""
        # Use layer type as strongest clue
        if 'layer_type' in entity:
            component_type = entity['layer_type'].capitalize()
            
            # Try to derive ID from position or handle
            if 'position' in entity:
                x, y = entity['position']
                # Generate ID based on position grid
                grid_size = 100  # Adjust based on drawing scale
                grid_x = int(x / grid_size)
                grid_y = int(y / grid_size)
                component_id = f"{grid_x}x{grid_y}"
                
                component = {
                    'type': component_type,
                    'id': component_id,
                    'handle': entity.get('handle'),
                    'position': entity.get('position'),
                    'inferred': True,
                    'source_entity': entity
                }
                
                # Check for duplicates
                if not self._is_duplicate_component(component):
                    self.detected_components.append(component)
                    return component
        
        # Check for rectangles which are often solar modules
        if entity.get('type') in ('POLYLINE', 'LWPOLYLINE', 'HATCH') and entity.get('is_rectangle', False):
            # Size-based classification
            area = entity.get('area', 0)
            if area > 0:
                component_type = None
                if 1 <= area <= 10:  # Small rectangle, likely a module
                    component_type = 'Module'
                elif 10 < area <= 100:  # Medium rectangle, could be combiner
                    component_type = 'Combiner'
                elif area > 100:  # Large rectangle, might be inverter or block
                    component_type = 'Inverter' if area < 1000 else 'Block'
                
                if component_type:
                    # Generate ID from position
                    x, y = entity.get('position')
                    grid_size = 100  # Adjust based on drawing scale
                    grid_x = int(x / grid_size)
                    grid_y = int(y / grid_size)
                    component_id = f"{grid_x}x{grid_y}"
                    
                    component = {
                        'type': component_type,
                        'id': component_id,
                        'handle': entity.get('handle'),
                        'position': entity.get('position'),
                        'inferred': True,
                        'source_entity': entity
                    }
                    
                    # Check for duplicates
                    if not self._is_duplicate_component(component):
                        self.detected_components.append(component)
                        return component
        
        # Check parent-child relationships to infer type
        if entity.get('handle') in self.parent_child_map:
            # This entity has children, could be a container type (Block, Inverter, Combiner)
            child_handles = self.parent_child_map[entity.get('handle')]
            
            # Look at entity size to determine type
            if 'area' in entity or ('width' in entity and 'height' in entity):
                area = entity.get('area', entity.get('width', 0) * entity.get('height', 0))
                
                if area > 1000000:  # Very large entity (arbitrary threshold)
                    component_type = 'Block'
                elif area > 100000:  # Large entity
                    component_type = 'Inverter'
                elif area > 10000:   # Medium entity
                    component_type = 'Combiner'
                else:
                    return None      # Too small, likely not a component
                
                # Generate ID based on position
                if 'position' in entity:
                    x, y = entity['position']
                    grid_size = 100  # Adjust based on drawing scale
                    grid_x = int(x / grid_size)
                    grid_y = int(y / grid_size)
                    component_id = f"{grid_x}x{grid_y}"
                    
                    component = {
                        'type': component_type,
                        'id': component_id,
                        'handle': entity.get('handle'),
                        'position': entity.get('position'),
                        'inferred': True,
                        'source_entity': entity
                    }
                    
                    # Check for duplicates
                    if not self._is_duplicate_component(component):
                        self.detected_components.append(component)
                        return component
        
        # If surroundings entities are provided, use them for additional context
        if surrounding_entities:
            # Find nearby text labels that might identify this entity
            if 'position' in entity:
                x, y = entity['position']
                text_entities = [e for e in surrounding_entities 
                               if e.get('type') in ('TEXT', 'MTEXT') and 'text' in e]
                
                # Find closest text entity
                closest_distance = float('inf')
                closest_text = None
                
                for text_entity in text_entities:
                    if 'position' in text_entity:
                        tx, ty = text_entity['position']
                        distance = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
                        
                        if distance < closest_distance and distance < 50:  # Max distance threshold
                            closest_distance = distance
                            closest_text = text_entity
                
                if closest_text:
                    # Try to identify component from the closest text
                    text = closest_text.get('text', '')
                    
                    # Check against naming patterns
                    for pattern_type, pattern in self.name_patterns.items():
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            component_type = pattern_type.capitalize()
                            component_id = self._extract_id_number(match)
                            
                            component = {
                                'type': component_type,
                                'id': component_id,
                                'text': text,
                                'handle': entity.get('handle'),
                                'position': entity.get('position'),
                                'inferred': True,
                                'source_entity': entity
                            }
                            
                            # Check for duplicates
                            if not self._is_duplicate_component(component):
                                self.detected_components.append(component)
                                return component
        
        return None

    def analyze_electrical_diagram(self, entities):
        """Analyze entities to detect electrical structure from single-line diagrams."""
        # Identify all text entities that might be labels
        labels = []
        for entity in entities:
            if entity.get('type') in ('TEXT', 'MTEXT') and 'text' in entity:
                labels.append(entity)
        
        # Identify key electrical components
        inverters = []
        transformers = []
        combiners = []
        strings = []
        modules = []
        
        # First pass: identify components from text labels
        for label in labels:
            # Check for inverter patterns
            if re.search(r'inverter|PCS|station|central', label['text'], re.IGNORECASE):
                # Look for nearby ID number
                id_match = re.search(r'(\d+)', label['text'])
                if id_match:
                    inverters.append({
                        'id': id_match.group(1),
                        'position': label['position'],
                        'handle': label['handle'],
                        'text': label['text']
                    })
            
            # Check for transformer patterns
            if re.search(r'transformer|xfmr|substation', label['text'], re.IGNORECASE):
                # Look for nearby ID number
                id_match = re.search(r'(\d+)', label['text'])
                if id_match:
                    transformers.append({
                        'id': id_match.group(1),
                        'position': label['position'],
                        'handle': label['handle'],
                        'text': label['text']
                    })
            
            # Check for combiner patterns
            if re.search(r'combiner|dcdb|smb|jb|junction', label['text'], re.IGNORECASE):
                # Look for nearby ID number
                id_match = re.search(r'(\d+)', label['text'])
                if id_match:
                    combiners.append({
                        'id': id_match.group(1),
                        'position': label['position'],
                        'handle': label['handle'],
                        'text': label['text']
                    })
            
            # Check for string patterns
            if re.search(r'string|str', label['text'], re.IGNORECASE):
                # Look for nearby ID number
                id_match = re.search(r'(\d+)', label['text'])
                if id_match:
                    strings.append({
                        'id': id_match.group(1),
                        'position': label['position'],
                        'handle': label['handle'],
                        'text': label['text']
                    })
            
            # Check for module patterns
            if re.search(r'module|panel|pv', label['text'], re.IGNORECASE):
                # Look for nearby ID number
                id_match = re.search(r'(\d+)', label['text'])
                if id_match:
                    modules.append({
                        'id': id_match.group(1),
                        'position': label['position'],
                        'handle': label['handle'],
                        'text': label['text']
                    })
        
        # Second pass: find potential component shapes (polylines, blocks, etc.)
        polyline_entities = [e for e in entities if e.get('type') in ('POLYLINE', 'LWPOLYLINE', 'HATCH')]
        block_entities = [e for e in entities if e.get('type') == 'INSERT']
        
        # Analyze polylines to find potential components
        for entity in polyline_entities:
            if 'area' not in entity or 'position' not in entity:
                continue
            
            area = entity['area']
            shape_type = None
            
            # Classify by size and shape
            if entity.get('is_rectangle', False):
                if area < 10:
                    shape_type = 'module'
                elif 10 <= area < 100:
                    shape_type = 'combiner'
                elif 100 <= area < 1000:
                    shape_type = 'inverter'
                else:
                    shape_type = 'structure'
            
            if shape_type:
                # Look for nearby text labels to identify
                closest_label = None
                closest_dist = float('inf')
                
                for label in labels:
                    if 'position' not in label:
                        continue
                    
                    dist = math.sqrt(
                        (entity['position'][0] - label['position'][0])**2 + 
                        (entity['position'][1] - label['position'][1])**2
                    )
                    
                    if dist < closest_dist and dist < 50:  # Threshold distance
                        closest_dist = dist
                        closest_label = label
                
                # If found a nearby label, use it to identify component
                if closest_label:
                    label_text = closest_label['text']
                    id_match = re.search(r'(\d+)', label_text)
                    component_id = id_match.group(1) if id_match else f"{int(entity['position'][0])}-{int(entity['position'][1])}"
                    
                    component = {
                        'id': component_id,
                        'position': entity['position'],
                        'handle': entity['handle'],
                        'text': closest_label['text'],
                        'area': area
                    }
                    
                    # Add to appropriate list
                    if shape_type == 'module' and 'module' in label_text.lower():
                        modules.append(component)
                    elif shape_type == 'combiner' and any(term in label_text.lower() for term in ['combiner', 'dcdb', 'smb', 'jb']):
                        combiners.append(component)
                    elif shape_type == 'inverter' and any(term in label_text.lower() for term in ['inverter', 'pcs', 'station']):
                        inverters.append(component)
        
        # Analyze lines to detect connections
        connections = []
        for entity in entities:
            if entity.get('type') == 'LINE' and 'start' in entity and 'end' in entity:
                start_x, start_y = entity['start']
                end_x, end_y = entity['end']
                
                # Find components near line endpoints
                start_component = None
                end_component = None
                
                # Check distance to inverters
                for inverter in inverters:
                    inv_x, inv_y = inverter['position']
                    start_dist = math.sqrt((start_x - inv_x)**2 + (start_y - inv_y)**2)
                    end_dist = math.sqrt((end_x - inv_x)**2 + (end_y - inv_y)**2)
                    
                    if start_dist < 20:  # Increased threshold distance
                        start_component = {'type': 'Inverter', 'id': inverter['id']}
                    if end_dist < 20:
                        end_component = {'type': 'Inverter', 'id': inverter['id']}
                
                # Check distance to transformers
                for transformer in transformers:
                    trans_x, trans_y = transformer['position']
                    start_dist = math.sqrt((start_x - trans_x)**2 + (start_y - trans_y)**2)
                    end_dist = math.sqrt((end_x - trans_x)**2 + (end_y - trans_y)**2)
                    
                    if start_dist < 20:
                        start_component = {'type': 'Transformer', 'id': transformer['id']}
                    if end_dist < 20:
                        end_component = {'type': 'Transformer', 'id': transformer['id']}
                
                # Check distance to combiners
                for combiner in combiners:
                    comb_x, comb_y = combiner['position']
                    start_dist = math.sqrt((start_x - comb_x)**2 + (start_y - comb_y)**2)
                    end_dist = math.sqrt((end_x - comb_x)**2 + (end_y - comb_y)**2)
                    
                    if start_dist < 20:
                        start_component = {'type': 'Combiner', 'id': combiner['id']}
                    if end_dist < 20:
                        end_component = {'type': 'Combiner', 'id': combiner['id']}
                
                # Check distance to strings
                for string in strings:
                    str_x, str_y = string['position']
                    start_dist = math.sqrt((start_x - str_x)**2 + (start_y - str_y)**2)
                    end_dist = math.sqrt((end_x - str_x)**2 + (end_y - str_y)**2)
                    
                    if start_dist < 20:
                        start_component = {'type': 'String', 'id': string['id']}
                    if end_dist < 20:
                        end_component = {'type': 'String', 'id': string['id']}
                
                if start_component and end_component:
                    # Determine connection direction (usually higher level to lower level)
                    # Inverter -> Combiner -> String -> Module
                    component_hierarchy = {
                        'Transformer': 4,
                        'Inverter': 3,
                        'Combiner': 2,
                        'String': 1,
                        'Module': 0
                    }
                    
                    # Ensure connection direction is from higher level to lower level
                    if component_hierarchy.get(start_component['type'], 0) < component_hierarchy.get(end_component['type'], 0):
                        # Swap direction
                        start_component, end_component = end_component, start_component
                    
                    connections.append({
                        'from': start_component,
                        'to': end_component,
                        'line': entity
                    })
        
        # Build a more complete electrical structure based on connections
        electrical_structure = {
            'components': {
                'Inverter': inverters,
                'Transformer': transformers,
                'Combiner': combiners,
                'String': strings,
                'Module': modules
            },
            'connections': connections
        }
        
        # Analyze connections to infer parent-child relationships
        for connection in connections:
            source_type = connection['from']['type']
            source_id = connection['from']['id']
            target_type = connection['to']['type']
            target_id = connection['to']['id']
            
            # Add parent-child relationship
            if source_type == 'Inverter' and target_type == 'Combiner':
                # Find the combiner in our list
                for combiner in combiners:
                    if combiner['id'] == target_id:
                        combiner['parent_id'] = source_id
                        combiner['parent_type'] = source_type
                        break
            elif source_type == 'Combiner' and target_type == 'String':
                # Find the string in our list
                for string in strings:
                    if string['id'] == target_id:
                        string['parent_id'] = source_id
                        string['parent_type'] = source_type
                        break
        
        return electrical_structure

class SolarEyeStructure(SiteStructurePort):
    """Implementation of the SiteStructurePort for SolarEye."""
    
    def __init__(self):
        self.structure = {
            'name': 'Unnamed Site',
            'capacity_mw': 0.0,
            'module_type': 'Unknown',
            'module_count': 0,
            'module_rated_w': 380,
            'location': '',
            'blocks': {}
        }
        
        # Keep track of component counts for statistics
        self.component_counts = {
            'Block': 0,
            'Inverter': 0,
            'Combiner': 0,
            'String': 0,
            'Module': 0,
            'Tracker': 0
        }
        
        # Track component hierarchy relationships
        self.hierarchy = defaultdict(list)
    
    def create_structure(self):
        """Create a new site structure."""
        self.structure = {
            'name': 'Unnamed Site',
            'capacity_mw': 0.0,
            'module_type': 'Unknown',
            'module_count': 0,
            'module_rated_w': 380,
            'location': '',
            'blocks': {}
        }
        
        # Reset component counts
        self.component_counts = {
            'Block': 0,
            'Inverter': 0,
            'Combiner': 0,
            'String': 0,
            'Module': 0,
            'Tracker': 0
        }
        
        # Reset hierarchy
        self.hierarchy = defaultdict(list)
        
        return self.structure
    
    def add_block(self, block_id, params=None):
        """Add a block to the structure."""
        params = params or {}
        
        # Create block if it doesn't exist
        if block_id not in self.structure['blocks']:
            self.structure['blocks'][block_id] = {
                'type': 'Block',
                'capacity_mw': params.get('capacity_mw', 0.0),
                'position': params.get('position', None),
                'components': {}
            }
            self.component_counts['Block'] += 1
        
        return self.structure['blocks'][block_id]
    
    def add_inverter(self, block_id, inverter_id, params=None):
        """Add an inverter to a block."""
        params = params or {}
        
        # Create block if it doesn't exist
        if block_id not in self.structure['blocks']:
            self.add_block(block_id)
        
        # Add inverter to block
        self.structure['blocks'][block_id]['components'][inverter_id] = {
            'type': 'Inverter',
            'capacity_mw': params.get('capacity_mw', 0.0),
            'position': params.get('position', None),
            'components': {}
        }
        
        # Update hierarchy
        self.hierarchy[block_id].append(inverter_id)
        
        self.component_counts['Inverter'] += 1
        
        return self.structure['blocks'][block_id]['components'][inverter_id]
    
    def add_combiner(self, block_id, inverter_id, combiner_id, params=None):
        """Add a combiner to an inverter."""
        params = params or {}
        
        # Create block if it doesn't exist
        if block_id not in self.structure['blocks']:
            self.add_block(block_id)
        
        # Create inverter if it doesn't exist
        if inverter_id not in self.structure['blocks'][block_id]['components']:
            self.add_inverter(block_id, inverter_id)
        
        # Add combiner to inverter
        self.structure['blocks'][block_id]['components'][inverter_id]['components'][combiner_id] = {
            'type': 'Combiner',
            'position': params.get('position', None),
            'components': {}
        }
        
        # Update hierarchy
        self.hierarchy[inverter_id].append(combiner_id)
        
        self.component_counts['Combiner'] += 1
        
        return self.structure['blocks'][block_id]['components'][inverter_id]['components'][combiner_id]
    
    def add_string(self, block_id, inverter_id, combiner_id, string_id, params=None):
        """Add a string to a combiner."""
        params = params or {}
        
        # Create block if it doesn't exist
        if block_id not in self.structure['blocks']:
            self.add_block(block_id)
        
        # Create inverter if it doesn't exist
        if inverter_id not in self.structure['blocks'][block_id]['components']:
            self.add_inverter(block_id, inverter_id)
        
        # Create combiner if it doesn't exist
        if combiner_id not in self.structure['blocks'][block_id]['components'][inverter_id]['components']:
            self.add_combiner(block_id, inverter_id, combiner_id)
        
        modules = params.get('modules', 20)
        
        # Add string to combiner
        self.structure['blocks'][block_id]['components'][inverter_id]['components'][combiner_id]['components'][string_id] = {
            'type': 'String',
            'modules': modules,
            'position': params.get('position', None)
        }
        
        # Update hierarchy
        self.hierarchy[combiner_id].append(string_id)
        
        # Update module count
        self.structure['module_count'] += modules
        self.component_counts['String'] += 1
        
        return self.structure['blocks'][block_id]['components'][inverter_id]['components'][combiner_id]['components'][string_id]
    
    def add_module(self, block_id, inverter_id, combiner_id, string_id, module_id, params=None):
        """Add a module to a string."""
        params = params or {}
        
        # Create block if it doesn't exist
        if block_id not in self.structure['blocks']:
            self.add_block(block_id)
        
        # Create inverter if it doesn't exist
        if inverter_id not in self.structure['blocks'][block_id]['components']:
            self.add_inverter(block_id, inverter_id)
        
        # Create combiner if it doesn't exist
        if combiner_id not in self.structure['blocks'][block_id]['components'][inverter_id]['components']:
            self.add_combiner(block_id, inverter_id, combiner_id)
        
        # Create string if it doesn't exist - we don't increment module count here as we're adding them explicitly
        if string_id not in self.structure['blocks'][block_id]['components'][inverter_id]['components'][combiner_id]['components']:
            self.structure['blocks'][block_id]['components'][inverter_id]['components'][combiner_id]['components'][string_id] = {
                'type': 'String',
                'modules': 0,
                'position': params.get('string_position', None),
                'modules_list': {}
            }
            self.component_counts['String'] += 1
        
        # Add module to string
        string = self.structure['blocks'][block_id]['components'][inverter_id]['components'][combiner_id]['components'][string_id]
        if 'modules_list' not in string:
            string['modules_list'] = {}
        
        string['modules_list'][module_id] = {
            'type': 'Module',
            'position': params.get('position', None),
            'rated_w': params.get('rated_w', self.structure['module_rated_w'])
        }
        
        # Update module counts
        string['modules'] += 1
        self.structure['module_count'] += 1
        self.component_counts['Module'] += 1
        
        # Update hierarchy
        self.hierarchy[string_id].append(module_id)
        
        return string['modules_list'][module_id]
    
    def add_tracker(self, block_id, tracker_id, params=None):
        """Add a tracker to a block."""
        params = params or {}
        
        # Create block if it doesn't exist
        if block_id not in self.structure['blocks']:
            self.add_block(block_id)
        
        # Add tracker to block
        if 'trackers' not in self.structure['blocks'][block_id]:
            self.structure['blocks'][block_id]['trackers'] = {}
        
        self.structure['blocks'][block_id]['trackers'][tracker_id] = {
            'type': 'Tracker',
            'position': params.get('position', None),
            'modules': params.get('modules', 0),
            'tables': params.get('tables', [])
        }
        
        # Update hierarchy
        self.hierarchy[block_id].append(tracker_id)
        
        self.component_counts['Tracker'] += 1
        
        return self.structure['blocks'][block_id]['trackers'][tracker_id]
    
    def find_parent(self, component_id):
        """Find the parent of a component."""
        for parent, children in self.hierarchy.items():
            if component_id in children:
                return parent
        return None
    
    def auto_structure_components(self, components):
        """Automatically structure components based on IDs and positions."""
        # First, sort components by type to ensure proper parent-child relationships
        components_by_type = {
            'Block': [],
            'Inverter': [],
            'Combiner': [],
            'String': [],
            'Module': [],
            'Tracker': []
        }
        
        for component in components:
            if component['type'] in components_by_type:
                components_by_type[component['type']].append(component)
        
        # Add blocks first
        for block in components_by_type['Block']:
            block_id = f"B{block['id']}"
            self.add_block(block_id, {
                'position': block.get('position')
            })
        
        # If no blocks found, create a default block
        if not components_by_type['Block']:
            self.add_block("B1")
        
        # Process inverters
        for inverter in components_by_type['Inverter']:
            inverter_id = f"I{inverter['id']}"
            
            # Try to determine parent block from ID
            block_match = re.search(r'B(\d+)', inverter.get('text', ''), re.IGNORECASE)
            if block_match:
                block_id = f"B{block_match.group(1)}"
            else:
                # Assign to nearest block based on position or first block
                block_id = self._find_nearest_block(inverter.get('position')) or "B1"
            
            # Ensure block exists
            if block_id not in self.structure['blocks']:
                self.add_block(block_id)
            
            # Add inverter
            self.add_inverter(block_id, inverter_id, {
                'position': inverter.get('position')
            })
        
        # Process combiners
        for combiner in components_by_type['Combiner']:
            combiner_id = f"C{combiner['id']}"
            
            # Try to determine parent block and inverter from ID
            block_id = None
            inverter_id = None
            
            # Check for block info in combiner text
            block_match = re.search(r'B(\d+)', combiner.get('text', ''), re.IGNORECASE)
            if block_match:
                block_id = f"B{block_match.group(1)}"
            
            # Check for inverter info in combiner text
            inverter_match = re.search(r'I(\d+)', combiner.get('text', ''), re.IGNORECASE)
            if inverter_match:
                inverter_id = f"I{inverter_match.group(1)}"
            
            # If parent info not found from text, use position to find nearest
            if not block_id or not inverter_id:
                nearest_inv, nearest_block = self._find_nearest_inverter(combiner.get('position'))
                if nearest_inv:
                    inverter_id = nearest_inv
                    block_id = nearest_block
                else:
                    # Fallback to first block and inverter
                    block_id = next(iter(self.structure['blocks'].keys()))
                    block = self.structure['blocks'][block_id]
                    if block['components']:
                        inverter_id = next(iter(block['components'].keys()))
                    else:
                        # Create inverter if needed
                        inverter_id = f"{block_id}-I1"
                        self.add_inverter(block_id, inverter_id)
            
            # Ensure block exists
            if block_id not in self.structure['blocks']:
                self.add_block(block_id)
            
            # Ensure inverter exists and is in correct format
            if not inverter_id.startswith(block_id):
                inverter_id = f"{block_id}-{inverter_id}"
            
            if inverter_id not in self.structure['blocks'][block_id]['components']:
                self.add_inverter(block_id, inverter_id)
            
            # Add combiner
            if not combiner_id.startswith(inverter_id):
                combiner_id = f"{inverter_id}-{combiner_id}"
            
            self.add_combiner(block_id, inverter_id, combiner_id, {
                'position': combiner.get('position')
            })
        
        # Process strings
        for string in components_by_type['String']:
            string_id = f"S{string['id']}"
            
            # Try to determine parent block, inverter, and combiner from ID
            block_id = None
            inverter_id = None
            combiner_id = None
            
            # Check for block info in string text
            block_match = re.search(r'B(\d+)', string.get('text', ''), re.IGNORECASE)
            if block_match:
                block_id = f"B{block_match.group(1)}"
            
            # Check for inverter info in string text
            inverter_match = re.search(r'I(\d+)', string.get('text', ''), re.IGNORECASE)
            if inverter_match:
                inverter_id = f"I{inverter_match.group(1)}"
            
            # Check for combiner info in string text
            combiner_match = re.search(r'C(\d+)', string.get('text', ''), re.IGNORECASE)
            if combiner_match:
                combiner_id = f"C{combiner_match.group(1)}"
            
            # If parent info not found from text, use position to find nearest
            if not block_id or not inverter_id or not combiner_id:
                nearest_comb, nearest_inv, nearest_block = self._find_nearest_combiner(string.get('position'))
                if nearest_comb:
                    combiner_id = nearest_comb
                    inverter_id = nearest_inv
                    block_id = nearest_block
                else:
                    # Fallback to first block, inverter, and combiner
                    block_id = next(iter(self.structure['blocks'].keys()))
                    block = self.structure['blocks'][block_id]
                    if block['components']:
                        inverter_id = next(iter(block['components'].keys()))
                        inverter = block['components'][inverter_id]
                        if inverter['components']:
                            combiner_id = next(iter(inverter['components'].keys()))
                        else:
                            # Create combiner if needed
                            combiner_id = f"{inverter_id}-C1"
                            self.add_combiner(block_id, inverter_id, combiner_id)
                    else:
                        # Create inverter and combiner if needed
                        inverter_id = f"{block_id}-I1"
                        self.add_inverter(block_id, inverter_id)
                        combiner_id = f"{inverter_id}-C1"
                        self.add_combiner(block_id, inverter_id, combiner_id)
            
            # Ensure block exists
            if block_id not in self.structure['blocks']:
                self.add_block(block_id)
            
            # Ensure inverter exists and is in correct format
            if not inverter_id.startswith(block_id):
                inverter_id = f"{block_id}-{inverter_id}"
            
            if inverter_id not in self.structure['blocks'][block_id]['components']:
                self.add_inverter(block_id, inverter_id)
            
            # Ensure combiner exists and is in correct format
            if not combiner_id.startswith(inverter_id):
                combiner_id = f"{inverter_id}-{combiner_id}"
            
            if combiner_id not in self.structure['blocks'][block_id]['components'][inverter_id]['components']:
                self.add_combiner(block_id, inverter_id, combiner_id)
            
            # Add string
            if not string_id.startswith(combiner_id):
                string_id = f"{combiner_id}-{string_id}"
            
            # Default 20 modules per string, can be refined later
            modules_per_string = 20
            
            self.add_string(block_id, inverter_id, combiner_id, string_id, {
                'modules': modules_per_string,
                'position': string.get('position')
            })
        
        # Process trackers
        for tracker in components_by_type['Tracker']:
            tracker_id = f"T{tracker['id']}"
            
            # Try to determine parent block from ID
            block_match = re.search(r'B(\d+)', tracker.get('text', ''), re.IGNORECASE)
            if block_match:
                block_id = f"B{block_match.group(1)}"
            else:
                # Assign to nearest block based on position or first block
                block_id = self._find_nearest_block(tracker.get('position')) or "B1"
            
            # Ensure block exists
            if block_id not in self.structure['blocks']:
                self.add_block(block_id)
            
            # Add tracker
            if not tracker_id.startswith(block_id):
                tracker_id = f"{block_id}-{tracker_id}"
            
            self.add_tracker(block_id, tracker_id, {
                'position': tracker.get('position'),
                'modules': 80  # Default modules per tracker, can be refined later
            })
        
        # Update site statistics
        self.calculate_site_statistics()
    
    def _find_nearest_block(self, position):
        """Find the nearest block to a position."""
        if not position:
            return None
        
        min_distance = float('inf')
        nearest_block = None
        
        for block_id, block in self.structure['blocks'].items():
            block_pos = block.get('position')
            if block_pos:
                distance = self._calculate_distance(position, block_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_block = block_id
        
        return nearest_block
    
    def _find_nearest_inverter(self, position):
        """Find the nearest inverter to a position."""
        if not position:
            return None, None
        
        min_distance = float('inf')
        nearest_inverter = None
        nearest_block = None
        
        for block_id, block in self.structure['blocks'].items():
            for inverter_id, inverter in block['components'].items():
                inverter_pos = inverter.get('position')
                if inverter_pos:
                    distance = self._calculate_distance(position, inverter_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_inverter = inverter_id
                        nearest_block = block_id
        
        return nearest_inverter, nearest_block
    
    def _find_nearest_combiner(self, position):
        """Find the nearest combiner to a position."""
        if not position:
            return None, None, None
        
        min_distance = float('inf')
        nearest_combiner = None
        nearest_inverter = None
        nearest_block = None
        
        for block_id, block in self.structure['blocks'].items():
            for inverter_id, inverter in block['components'].items():
                for combiner_id, combiner in inverter['components'].items():
                    combiner_pos = combiner.get('position')
                    if combiner_pos:
                        distance = self._calculate_distance(position, combiner_pos)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_combiner = combiner_id
                            nearest_inverter = inverter_id
                            nearest_block = block_id
        
        return nearest_combiner, nearest_inverter, nearest_block
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def calculate_site_statistics(self):
        """Calculate site statistics based on components."""
        total_modules = self.structure['module_count']
        
        # Calculate capacity based on module count and rating
        module_rated_w = self.structure['module_rated_w']
        capacity_mw = (total_modules * module_rated_w) / 1000000
        
        self.structure['capacity_mw'] = capacity_mw
        
        # Calculate average modules per string
        if self.component_counts['String'] > 0:
            avg_modules_per_string = total_modules / self.component_counts['String']
        else:
            avg_modules_per_string = 0
        
        self.structure['avg_modules_per_string'] = avg_modules_per_string
        
        # Calculate average strings per combiner
        if self.component_counts['Combiner'] > 0:
            avg_strings_per_combiner = self.component_counts['String'] / self.component_counts['Combiner']
        else:
            avg_strings_per_combiner = 0
        
        self.structure['avg_strings_per_combiner'] = avg_strings_per_combiner
        
        # Calculate average combiners per inverter
        if self.component_counts['Inverter'] > 0:
            avg_combiners_per_inverter = self.component_counts['Combiner'] / self.component_counts['Inverter']
        else:
            avg_combiners_per_inverter = 0
        
        self.structure['avg_combiners_per_inverter'] = avg_combiners_per_inverter
        
        return {
            'total_modules': total_modules,
            'capacity_mw': capacity_mw,
            'component_counts': self.component_counts,
            'avg_modules_per_string': avg_modules_per_string,
            'avg_strings_per_combiner': avg_strings_per_combiner,
            'avg_combiners_per_inverter': avg_combiners_per_inverter
        }
    
    def export_structure(self, format_type, file_path=None):
        """Export the structure to the specified format."""
        if format_type.lower() == 'json':
            # Format for SolarEye JSON
            output = json.dumps(self.structure, indent=2)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(output)
            
            return output
            
        elif format_type.lower() == 'csv':
            # Format for SolarEye CSV
            lines = ["component_type,id,name,parent_id,block_id,inverter_id,combiner_id,module_count"]
            
            # Add site
            lines.append(f"Site,{self.structure['name']},{self.structure['name']},,,,")
            
            # Add blocks
            for block_id, block in self.structure['blocks'].items():
                lines.append(f"Block,{block_id},{block_id},{self.structure['name']},,,")
                
                # Add inverters
                for inverter_id, inverter in block['components'].items():
                    lines.append(f"Inverter,{inverter_id},{inverter_id},{block_id},{block_id},,")
                    
                    # Add combiners
                    for combiner_id, combiner in inverter['components'].items():
                        lines.append(f"Combiner,{combiner_id},{combiner_id},{inverter_id},{block_id},{inverter_id},")
                        
                        # Add strings
                        for string_id, string in combiner['components'].items():
                            lines.append(f"String,{string_id},{string_id},{combiner_id},{block_id},{inverter_id},{combiner_id},{string['modules']}")
                
                # Add trackers if present
                if 'trackers' in block:
                    for tracker_id, tracker in block['trackers'].items():
                        lines.append(f"Tracker,{tracker_id},{tracker_id},{block_id},{block_id},,{tracker.get('modules', 0)}")
            
            output = "\n".join(lines)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(output)
            
            return output
            
        elif format_type.lower() == 'excel' and EXCEL_AVAILABLE:
            # Export as Excel file
            if file_path:
                workbook = xlsxwriter.Workbook(file_path)
                
                # Add formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D0D0D0',
                    'border': 1
                })
                
                cell_format = workbook.add_format({
                    'border': 1
                })
                
                # Create Site sheet
                site_sheet = workbook.add_worksheet('Site')
                
                # Add header
                site_sheet.write(0, 0, 'Property', header_format)
                site_sheet.write(0, 1, 'Value', header_format)
                
                # Add site properties
                site_sheet.write(1, 0, 'name', cell_format)
                site_sheet.write(1, 1, self.structure['name'], cell_format)
                
                site_sheet.write(2, 0, 'capacity_mw', cell_format)
                site_sheet.write(2, 1, self.structure['capacity_mw'], cell_format)
                
                site_sheet.write(3, 0, 'module_type', cell_format)
                site_sheet.write(3, 1, self.structure['module_type'], cell_format)
                
                site_sheet.write(4, 0, 'module_count', cell_format)
                site_sheet.write(4, 1, self.structure['module_count'], cell_format)
                
                site_sheet.write(5, 0, 'module_rated_w', cell_format)
                site_sheet.write(5, 1, self.structure['module_rated_w'], cell_format)
                
                site_sheet.write(6, 0, 'location', cell_format)
                site_sheet.write(6, 1, self.structure['location'], cell_format)
                
                # Add statistics
                site_sheet.write(8, 0, 'Statistics', header_format)
                
                row = 9
                for component_type, count in self.component_counts.items():
                    site_sheet.write(row, 0, f"{component_type} Count", cell_format)
                    site_sheet.write(row, 1, count, cell_format)
                    row += 1
                
                site_sheet.write(row, 0, 'Avg Modules per String', cell_format)
                site_sheet.write(row, 1, self.structure.get('avg_modules_per_string', 0), cell_format)
                row += 1
                
                site_sheet.write(row, 0, 'Avg Strings per Combiner', cell_format)
                site_sheet.write(row, 1, self.structure.get('avg_strings_per_combiner', 0), cell_format)
                row += 1
                
                site_sheet.write(row, 0, 'Avg Combiners per Inverter', cell_format)
                site_sheet.write(row, 1, self.structure.get('avg_combiners_per_inverter', 0), cell_format)
                
                # Create Components sheet
                comp_sheet = workbook.add_worksheet('Components')
                
                # Add header
                cols = ['component_type', 'id', 'name', 'parent_id', 'block_id', 
                       'inverter_id', 'combiner_id', 'module_count']
                
                for i, col in enumerate(cols):
                    comp_sheet.write(0, i, col, header_format)
                
                # Add site
                row = 1
                comp_sheet.write(row, 0, 'Site', cell_format)
                comp_sheet.write(row, 1, self.structure['name'], cell_format)
                comp_sheet.write(row, 2, self.structure['name'], cell_format)
                for i in range(3, 8):
                    comp_sheet.write(row, i, '', cell_format)
                row += 1
                
                # Add blocks
                for block_id, block in self.structure['blocks'].items():
                    comp_sheet.write(row, 0, 'Block', cell_format)
                    comp_sheet.write(row, 1, block_id, cell_format)
                    comp_sheet.write(row, 2, block_id, cell_format)
                    comp_sheet.write(row, 3, self.structure['name'], cell_format)
                    for i in range(4, 8):
                        comp_sheet.write(row, i, '', cell_format)
                    row += 1
                    
                    # Add inverters
                    for inverter_id, inverter in block['components'].items():
                        comp_sheet.write(row, 0, 'Inverter', cell_format)
                        comp_sheet.write(row, 1, inverter_id, cell_format)
                        comp_sheet.write(row, 2, inverter_id, cell_format)
                        comp_sheet.write(row, 3, block_id, cell_format)
                        comp_sheet.write(row, 4, block_id, cell_format)
                        for i in range(5, 8):
                            comp_sheet.write(row, i, '', cell_format)
                        row += 1
                        
                        # Add combiners
                        for combiner_id, combiner in inverter['components'].items():
                            comp_sheet.write(row, 0, 'Combiner', cell_format)
                            comp_sheet.write(row, 1, combiner_id, cell_format)
                            comp_sheet.write(row, 2, combiner_id, cell_format)
                            comp_sheet.write(row, 3, inverter_id, cell_format)
                            comp_sheet.write(row, 4, block_id, cell_format)
                            comp_sheet.write(row, 5, inverter_id, cell_format)
                            for i in range(6, 8):
                                comp_sheet.write(row, i, '', cell_format)
                            row += 1
                            
                            # Add strings
                            for string_id, string in combiner['components'].items():
                                comp_sheet.write(row, 0, 'String', cell_format)
                                comp_sheet.write(row, 1, string_id, cell_format)
                                comp_sheet.write(row, 2, string_id, cell_format)
                                comp_sheet.write(row, 3, combiner_id, cell_format)
                                comp_sheet.write(row, 4, block_id, cell_format)
                                comp_sheet.write(row, 5, inverter_id, cell_format)
                                comp_sheet.write(row, 6, combiner_id, cell_format)
                                comp_sheet.write(row, 7, string['modules'], cell_format)
                                row += 1
                    
                    # Add trackers if present
                    if 'trackers' in block:
                        for tracker_id, tracker in block['trackers'].items():
                            comp_sheet.write(row, 0, 'Tracker', cell_format)
                            comp_sheet.write(row, 1, tracker_id, cell_format)
                            comp_sheet.write(row, 2, tracker_id, cell_format)
                            comp_sheet.write(row, 3, block_id, cell_format)
                            comp_sheet.write(row, 4, block_id, cell_format)
                            for i in range(5, 7):
                                comp_sheet.write(row, i, '', cell_format)
                            comp_sheet.write(row, 7, tracker.get('modules', 0), cell_format)
                            row += 1
                
                workbook.close()
                return file_path
                
            return "Excel export requires a file path"
            
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

class CADAnalyzerMainWindow(QMainWindow):
    """Main window for CAD Analyzer application."""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("SolarEye - CAD Structure Analyzer")
        self.setMinimumSize(1000, 700)
        
        # Initialize data
        self.cad_data = None
        self.cad_entities = []
        self.cad_blocks = []
        self.detected_components = []
        self.site_structure = SolarEyeStructure()
        self.cad_adapter = None
        
        # Initialize visualization properties
        self.visualization_scale = 1.0
        self.visualization_offset = (0, 0)
        self.selected_component = None
        
        # Create UI
        self._create_ui()
        self._create_menus()
        
        # Set initial state
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def _create_ui(self):
        """Create the main UI layout."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        toolbar_layout = QHBoxLayout()
        
        load_cad_button = QPushButton("Load CAD File")
        load_cad_button.clicked.connect(self.on_load_cad)
        toolbar_layout.addWidget(load_cad_button)
        
        analyze_button = QPushButton("Analyze CAD")
        analyze_button.clicked.connect(self.on_analyze_cad)
        toolbar_layout.addWidget(analyze_button)
        
        export_button = QPushButton("Export Structure")
        export_button.clicked.connect(self.on_export)
        toolbar_layout.addWidget(export_button)
        
        toolbar_layout.addStretch()
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        toolbar_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(toolbar_layout)
        
        # Create splitter for main area
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Components tree and info
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Components tab
        components_tab = QWidget()
        components_layout = QVBoxLayout(components_tab)
        
        # Components tree
        components_group = QGroupBox("Detected Components")
        components_tree_layout = QVBoxLayout(components_group)
        
        self.components_tree = QTreeWidget()
        self.components_tree.setHeaderLabels(["Component", "ID", "Type"])
        self.components_tree.setColumnWidth(0, 150)
        self.components_tree.setColumnWidth(1, 80)
        self.components_tree.itemClicked.connect(self.on_component_selected)
        components_tree_layout.addWidget(self.components_tree)
        
        components_layout.addWidget(components_group)
        
        # Site statistics
        stats_group = QGroupBox("Site Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QLabel("No CAD file loaded")
        stats_layout.addWidget(self.stats_text)
        
        components_layout.addWidget(stats_group)
        
        # Structure tab
        structure_tab = QWidget()
        structure_layout = QVBoxLayout(structure_tab)
        
        # Structure tree
        structure_group = QGroupBox("Site Structure")
        structure_tree_layout = QVBoxLayout(structure_group)
        
        self.structure_tree = QTreeWidget()
        self.structure_tree.setHeaderLabels(["Component", "ID", "Details"])
        self.structure_tree.setColumnWidth(0, 150)
        self.structure_tree.setColumnWidth(1, 80)
        self.structure_tree.itemClicked.connect(self.on_structure_item_selected)
        structure_tree_layout.addWidget(self.structure_tree)
        
        structure_layout.addWidget(structure_group)
        
        # Structure settings
        settings_group = QGroupBox("Structure Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.site_name_edit = QLineEdit("Solar Site")
        settings_layout.addRow("Site Name:", self.site_name_edit)
        
        self.module_type_edit = QLineEdit("Standard 380W")
        settings_layout.addRow("Module Type:", self.module_type_edit)
        
        self.module_power_edit = QLineEdit("380")
        settings_layout.addRow("Module Power (W):", self.module_power_edit)
        
        structure_layout.addWidget(settings_group)
        
        # CAD Visualization Options tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        viz_options_group = QGroupBox("Visualization Options")
        viz_options_layout = QVBoxLayout(viz_options_group)
        
        # Color coding options
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color Code:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Component Type", "Layer", "None"])
        self.color_combo.currentIndexChanged.connect(self.on_visualization_option_changed)
        color_layout.addWidget(self.color_combo)
        viz_options_layout.addLayout(color_layout)
        
        # Show/hide options
        show_options_layout = QVBoxLayout()
        
        self.show_blocks_check = QCheckBox("Show Blocks")
        self.show_blocks_check.setChecked(True)
        self.show_blocks_check.stateChanged.connect(self.on_visualization_option_changed)
        show_options_layout.addWidget(self.show_blocks_check)
        
        self.show_inverters_check = QCheckBox("Show Inverters")
        self.show_inverters_check.setChecked(True)
        self.show_inverters_check.stateChanged.connect(self.on_visualization_option_changed)
        show_options_layout.addWidget(self.show_inverters_check)
        
        self.show_combiners_check = QCheckBox("Show Combiners")
        self.show_combiners_check.setChecked(True)
        self.show_combiners_check.stateChanged.connect(self.on_visualization_option_changed)
        show_options_layout.addWidget(self.show_combiners_check)
        
        self.show_strings_check = QCheckBox("Show Strings")
        self.show_strings_check.setChecked(True)
        self.show_strings_check.stateChanged.connect(self.on_visualization_option_changed)
        show_options_layout.addWidget(self.show_strings_check)
        
        self.show_modules_check = QCheckBox("Show Modules")
        self.show_modules_check.setChecked(True)
        self.show_modules_check.stateChanged.connect(self.on_visualization_option_changed)
        show_options_layout.addWidget(self.show_modules_check)
        
        self.show_trackers_check = QCheckBox("Show Trackers")
        self.show_trackers_check.setChecked(True)
        self.show_trackers_check.stateChanged.connect(self.on_visualization_option_changed)
        show_options_layout.addWidget(self.show_trackers_check)
        
        viz_options_layout.addLayout(show_options_layout)
        
        # Visualization controls
        viz_controls_layout = QHBoxLayout()
        
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.on_zoom_in)
        viz_controls_layout.addWidget(zoom_in_button)
        
        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.on_zoom_out)
        viz_controls_layout.addWidget(zoom_out_button)
        
        reset_view_button = QPushButton("Reset View")
        reset_view_button.clicked.connect(self.on_reset_view)
        viz_controls_layout.addWidget(reset_view_button)
        
        viz_options_layout.addLayout(viz_controls_layout)
        viz_layout.addWidget(viz_options_group)
        
        # DWG import options
        dwg_options_group = QGroupBox("DWG Import Options")
        dwg_options_layout = QVBoxLayout(dwg_options_group)
        
        conversion_layout = QHBoxLayout()
        conversion_layout.addWidget(QLabel("Preferred Conversion Method:"))
        self.conversion_combo = QComboBox()
        self.conversion_combo.addItems([tool[0] for tool in DWGConverterUtil.find_conversion_tools()])
        conversion_layout.addWidget(self.conversion_combo)
        dwg_options_layout.addLayout(conversion_layout)
        
        viz_layout.addWidget(dwg_options_group)
        viz_layout.addStretch()
        
        # Add tabs
        tabs.addTab(components_tab, "Components")
        tabs.addTab(structure_tab, "Structure")
        tabs.addTab(viz_tab, "Visualization Options")
        left_layout.addWidget(tabs)
        
        # Right panel - CAD visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # CAD visualization area
        cad_group = QGroupBox("CAD Visualization")
        cad_layout = QVBoxLayout(cad_group)
        
        self.cad_view = QLabel("No CAD file loaded")
        self.cad_view.setAlignment(Qt.AlignCenter)
        self.cad_view.setStyleSheet("background-color: #f0f0f0;")
        self.cad_view.setMinimumSize(500, 400)
        cad_layout.addWidget(self.cad_view)
        
        right_layout.addWidget(cad_group)
        
        # Details panel
        details_group = QGroupBox("Component Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QLabel("Select a component to view details")
        details_layout.addWidget(self.details_text)
        
        right_layout.addWidget(details_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 700])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
    
    def _create_menus(self):
        """Create application menus."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        load_cad_action = file_menu.addAction("&Load CAD File...")
        load_cad_action.triggered.connect(self.on_load_cad)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)
        
        # Analysis menu
        analysis_menu = menu_bar.addMenu("&Analysis")
        
        analyze_action = analysis_menu.addAction("&Analyze CAD")
        analyze_action.triggered.connect(self.on_analyze_cad)
        
        # Export menu
        export_menu = menu_bar.addMenu("&Export")
        
        export_json_action = export_menu.addAction("Export to &JSON...")
        export_json_action.triggered.connect(lambda: self.on_export('json'))
        
        export_csv_action = export_menu.addAction("Export to &CSV...")
        export_csv_action.triggered.connect(lambda: self.on_export('csv'))
        
        if EXCEL_AVAILABLE:
            export_excel_action = export_menu.addAction("Export to &Excel...")
            export_excel_action.triggered.connect(lambda: self.on_export('excel'))
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self.on_about)
    
    def on_load_cad(self):
        """Load CAD file."""
        # Create a more comprehensive file filter including DWG
        file_filter = "All supported files (*.dxf *.dwg);;DXF files (*.dxf);;DWG files (*.dwg);;All files (*.*)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CAD File", "", file_filter
        )
        
        if not file_path:
            return
        
        try:
            self.status_bar.showMessage(f"Loading CAD file: {os.path.basename(file_path)}...")
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            # Initialize CAD adapter based on file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.dxf', '.dwg']:
                if EZDXF_AVAILABLE:
                    # Use the enhanced DWG adapter with dialog support
                    self.cad_adapter = EnhancedDWGEzdxfAdapter(parent_widget=self)
                else:
                    QMessageBox.critical(self, "Error", "No CAD library available. Please install ezdxf.")
                    return
            else:
                QMessageBox.critical(self, "Error", f"Unsupported file format: {file_ext}")
                return
            
            # Load CAD file
            self.cad_data = self.cad_adapter.read_file(file_path)
            self.progress_bar.setValue(40)
            
            if not self.cad_data:
                raise ValueError("Failed to load CAD file - no data returned")
            
            # Extract entities
            self.cad_entities = self.cad_adapter.extract_entities(self.cad_data)
            self.progress_bar.setValue(70)
            
            # Extract blocks
            self.cad_blocks = self.cad_adapter.extract_blocks(self.cad_data)
            self.progress_bar.setValue(90)
            
            # Update UI
            self.status_bar.showMessage(f"Loaded CAD file: {os.path.basename(file_path)}")
            self.progress_bar.setValue(100)
            
            # Display basic CAD info
            self.update_cad_info()
            
            # Clear previous component data
            self.components_tree.clear()
            self.structure_tree.clear()
            self.detected_components = []
            
            # Display CAD visualization
            self.visualize_cad()
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error loading CAD file: {str(e)}\n{error_details}")
            QMessageBox.critical(self, "Load Error", f"Error loading CAD file: {str(e)}")
            self.status_bar.showMessage("Error loading CAD file")
            self.progress_bar.setValue(0)
    
    def update_cad_info(self):
        """Update CAD info display."""
        if not self.cad_data or not self.cad_entities:
            return
        
        # Display basic statistics
        stats_text = (
            f"Total entities: {len(self.cad_entities)}\n"
            f"Block definitions: {len(self.cad_blocks)}\n"
        )
        
        # Count entity types
        entity_types = defaultdict(int)
        for entity in self.cad_entities:
            entity_types[entity.get('type', 'Unknown')] += 1
        
        stats_text += "\nEntity types:\n"
        for entity_type, count in entity_types.items():
            stats_text += f"- {entity_type}: {count}\n"
        
        self.stats_text.setText(stats_text)
    
    def on_analyze_cad(self):
        """Analyze CAD file to detect components."""
        if not self.cad_data or not self.cad_entities:
            QMessageBox.warning(self, "Warning", "No CAD file loaded")
            return
        
        try:
            self.status_bar.showMessage("Analyzing CAD file...")
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            # Use existing CAD adapter if available
            if not self.cad_adapter:
                if EZDXF_AVAILABLE:
                    self.cad_adapter = EnhancedDWGEzdxfAdapter(parent_widget=self)
                else:
                    QMessageBox.critical(self, "Error", "No CAD library available. Please install ezdxf.")
                    return
            
            # Clear previous detected components
            self.detected_components = []
            
            # Process entities to identify components
            total_entities = len(self.cad_entities)
            self.progress_bar.setValue(20)
            
            # Detect components
            chunk_size = 100  # Process in chunks to update progress
            for i in range(0, total_entities, chunk_size):
                chunk = self.cad_entities[i:min(i+chunk_size, total_entities - i)]
                chunk_entities = self.cad_entities[i:i+chunk_size]
                
                for entity in chunk_entities:
                    component = self.cad_adapter.identify_component(entity, self.cad_entities)
                    if component:
                        self.detected_components.append(component)
                
                # Update progress
                progress = 20 + (i / total_entities) * 60
                self.progress_bar.setValue(progress)
                QApplication.processEvents()
            
            # Group components by type for better organization
            components_by_type = defaultdict(list)
            for component in self.detected_components:
                components_by_type[component['type']].append(component)
            
            # Try to detect electrical structure
            try:
                if hasattr(self.cad_adapter, 'analyze_electrical_diagram'):
                    electrical_structure = self.cad_adapter.analyze_electrical_diagram(self.cad_entities)
                    
                    # Merge additional detected components
                    for comp_type, components in electrical_structure['components'].items():
                        for comp in components:
                            if 'id' in comp and not any(c.get('id') == comp['id'] and c.get('type') == comp_type for c in self.detected_components):
                                self.detected_components.append({
                                    'type': comp_type,
                                    'id': comp['id'],
                                    'position': comp.get('position'),
                                    'text': comp.get('text', '')
                                })
                                components_by_type[comp_type].append(comp)
            except Exception as e:
                logger.warning(f"Electrical structure analysis failed: {str(e)}")
            
            self.progress_bar.setValue(85)
            
            # Create site structure from detected components
            self.site_structure = SolarEyeStructure()
            self.site_structure.create_structure()
            
            # Update site info from UI
            self.site_structure.structure['name'] = self.site_name_edit.text()
            self.site_structure.structure['module_type'] = self.module_type_edit.text()
            try:
                self.site_structure.structure['module_rated_w'] = int(self.module_power_edit.text())
            except ValueError:
                self.site_structure.structure['module_rated_w'] = 380  # Default if invalid
            
            # Auto-structure components
            self.site_structure.auto_structure_components(self.detected_components)
            
            # Calculate site statistics
            stats = self.site_structure.calculate_site_statistics()
            
            self.progress_bar.setValue(95)
            
            # Update UI
            self.update_components_tree(components_by_type)
            self.update_structure_tree()
            self.update_site_statistics(stats)
            
            # Visualize with detected components highlighted
            self.visualize_cad(highlight_components=True)
            
            self.status_bar.showMessage(f"Analysis complete: {len(self.detected_components)} components detected")
            self.progress_bar.setValue(100)
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error analyzing CAD file: {str(e)}\n{error_details}")
            QMessageBox.critical(self, "Analysis Error", f"Error analyzing CAD file: {str(e)}")
            self.status_bar.showMessage("Error analyzing CAD file")
            self.progress_bar.setValue(0)
    
    def update_components_tree(self, components_by_type):
        """Update components tree with detected components."""
        self.components_tree.clear()
        
        # Add root item
        root = QTreeWidgetItem(self.components_tree, ["Components", "", ""])
        root.setExpanded(True)
        
        # Add component types as children
        for comp_type, components in components_by_type.items():
            type_item = QTreeWidgetItem(root, [comp_type, str(len(components)), ""])
            
            # Add components as children of type
            for component in components:
                comp_name = f"{comp_type} {component['id']}"
                comp_item = QTreeWidgetItem(type_item, [comp_name, component['id'], comp_type])
                comp_item.setData(0, Qt.UserRole, component)  # Store full component data
        
        self.components_tree.expandAll()
    
    def update_structure_tree(self):
        """Update structure tree with site structure."""
        self.structure_tree.clear()
        
        # Add site as root
        site_name = self.site_structure.structure['name']
        site_details = f"Capacity: {self.site_structure.structure['capacity_mw']:.2f} MW"
        root = QTreeWidgetItem(self.structure_tree, [site_name, "", site_details])
        root.setExpanded(True)
        
        # Add blocks as children
        for block_id, block in self.site_structure.structure['blocks'].items():
            block_details = f"Type: Block"
            block_item = QTreeWidgetItem(root, [block_id, block_id, block_details])
            
            # Add inverters as children of block
            for inverter_id, inverter in block['components'].items():
                inverter_details = f"Type: Inverter"
                inverter_item = QTreeWidgetItem(block_item, [inverter_id, inverter_id.split('-')[-1], inverter_details])
                
                # Add combiners as children of inverter
                for combiner_id, combiner in inverter['components'].items():
                    combiner_details = f"Type: Combiner"
                    combiner_item = QTreeWidgetItem(inverter_item, [combiner_id, combiner_id.split('-')[-1], combiner_details])
                    
                    # Add strings as children of combiner
                    for string_id, string in combiner['components'].items():
                        string_details = f"Modules: {string['modules']}"
                        string_item = QTreeWidgetItem(combiner_item, [string_id, string_id.split('-')[-1], string_details])
            
            # Add trackers as children of block (if present)
            if 'trackers' in block:
                for tracker_id, tracker in block['trackers'].items():
                    tracker_details = f"Type: Tracker, Modules: {tracker.get('modules', 0)}"
                    tracker_item = QTreeWidgetItem(block_item, [tracker_id, tracker_id.split('-')[-1], tracker_details])
        
        self.structure_tree.expandToDepth(1)  # Expand to show blocks
    
    def update_site_statistics(self, stats):
        """Update site statistics display."""
        stats_text = (
            f"Site Name: {self.site_structure.structure['name']}\n"
            f"Capacity: {stats['capacity_mw']:.2f} MW\n"
            f"Module Type: {self.site_structure.structure['module_type']}\n"
            f"Module Rating: {self.site_structure.structure['module_rated_w']} W\n"
            f"Total Modules: {stats['total_modules']}\n\n"
            f"Component Counts:\n"
        )
        
        for comp_type, count in stats['component_counts'].items():
            stats_text += f"- {comp_type}: {count}\n"
        
        stats_text += f"\nAvg. Modules per String: {stats['avg_modules_per_string']:.1f}\n"
        stats_text += f"Avg. Strings per Combiner: {stats['avg_strings_per_combiner']:.1f}\n"
        stats_text += f"Avg. Combiners per Inverter: {stats['avg_combiners_per_inverter']:.1f}\n"
        
        self.stats_text.setText(stats_text)
    
    def on_component_selected(self, item, column):
        """Handle component selection in the tree."""
        # Get component data
        component_data = item.data(0, Qt.UserRole)
        if not component_data:
            self.details_text.setText("No component data available")
            return
        
        # Update details display
        details = f"Component Type: {component_data['type']}\n"
        details += f"ID: {component_data['id']}\n"
        
        if 'position' in component_data:
            details += f"Position: ({component_data['position'][0]:.1f}, {component_data['position'][1]:.1f})\n"
        
        if 'text' in component_data:
            details += f"Label Text: {component_data['text']}\n"
        
        if 'source_entity' in component_data:
            source = component_data['source_entity']
            details += f"Source Entity Type: {source.get('type', 'Unknown')}\n"
            details += f"Layer: {source.get('layer', 'Unknown')}\n"
            
            if 'block_name' in source:
                details += f"Block: {source['block_name']}\n"
        
        self.details_text.setText(details)
        
        # Highlight selected component in visualization
        self.selected_component = component_data
        self.visualize_cad(highlight_components=True)
    
    def on_structure_item_selected(self, item, column):
        """Handle structure item selection in the tree."""
        # Parse item id and details
        item_id = item.text(1)
        if not item_id:
            self.details_text.setText("No component data available")
            return
        
        # Find corresponding component in detected components
        component = None
        for comp in self.detected_components:
            comp_id_part = comp['id']
            if item_id == comp_id_part or item.text(0).endswith(comp_id_part):
                component = comp
                break
        
        if component:
            # Update details display
            details = f"Component Type: {component['type']}\n"
            details += f"ID: {component['id']}\n"
            
            if 'position' in component:
                details += f"Position: ({component['position'][0]:.1f}, {component['position'][1]:.1f})\n"
            
            if 'text' in component:
                details += f"Label Text: {component['text']}\n"
            
            self.details_text.setText(details)
            
            # Highlight selected component in visualization
            self.selected_component = component
            self.visualize_cad(highlight_components=True)
        else:
            # Show structure info from site structure
            comp_path = []
            current = item
            while current:
                if current.text(0):
                    comp_path.insert(0, current.text(0))
                current = current.parent()
            
            details = f"Component: {' > '.join(comp_path)}\n"
            details += f"ID: {item.text(0)}\n"
            details += f"Details: {item.text(2)}\n"
            
            self.details_text.setText(details)
            
            # Clear selected component
            self.selected_component = None
            self.visualize_cad(highlight_components=True)
    
    def visualize_cad(self, highlight_components=False):
        """Visualize CAD data."""
        if not self.cad_entities:
            return
        
        try:
            # Create a QImage for visualization
            width, height = 800, 600
            image = QImage(width, height, QImage.Format_RGB32)
            image.fill(QColor(240, 240, 240))
            
            # Create painter
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Calculate bounds if not available
            if not hasattr(self.cad_adapter, 'bounds') or not self.cad_adapter.bounds:
                min_x, min_y = float('inf'), float('inf')
                max_x, max_y = float('-inf'), float('-inf')
                
                for entity in self.cad_entities:
                    if 'position' in entity:
                        x, y = entity['position']
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                
                if min_x == float('inf'):
                    min_x, min_y = 0, 0
                if max_x == float('-inf'):
                    max_x, max_y = 1000, 1000
                
                bounds = (min_x, min_y, max_x, max_y)
            else:
                bounds = self.cad_adapter.bounds
            
            # Calculate scaling factors based on bounds
            min_x, min_y, max_x, max_y = bounds
            margin = 50  # Margin in pixels
            
            # Calculate scaling based on visualization area size and scale factor
            scale_x = (width - 2 * margin) / (max_x - min_x) * self.visualization_scale
            scale_y = (height - 2 * margin) / (max_y - min_y) * self.visualization_scale
            scale = min(scale_x, scale_y)  # Use minimum scale to ensure everything fits
            
            # Calculate offset to center the visualization
            offset_x = margin - min_x * scale + self.visualization_offset[0]
            offset_y = margin - min_y * scale + self.visualization_offset[1]
            
            # Draw entities
            for entity in self.cad_entities:
                # Determine color based on layer or type
                if self.color_combo.currentText() == "Layer":
                    # Color by layer
                    layer = entity.get('layer', 'default')
                    color = self._get_color_for_layer(layer)
                elif self.color_combo.currentText() == "Component Type":
                    # Try to determine component type
                    if 'component_hint' in entity:
                        color = self._get_color_for_component_type(entity['component_hint'])
                    elif 'layer_type' in entity:
                        color = self._get_color_for_component_type(entity['layer_type'])
                    else:
                        # Default color by entity type
                        color = self._get_color_for_entity_type(entity.get('type', 'unknown'))
                else:
                    # Default - color by entity type
                    color = self._get_color_for_entity_type(entity.get('type', 'unknown'))
                
                painter.setPen(QPen(QColor(*color), 1))
                
                # Draw based on entity type
                if entity.get('type') in ('TEXT', 'MTEXT') and 'position' in entity:
                    x, y = entity['position']
                    screen_x = x * scale + offset_x
                    screen_y = y * scale + offset_y
                    painter.drawText(QPoint(screen_x, screen_y), entity.get('text', ''))
                
                elif entity.get('type') == 'LINE' and 'start' in entity and 'end' in entity:
                    x1, y1 = entity['start']
                    x2, y2 = entity['end']
                    screen_x1 = x1 * scale + offset_x
                    screen_y1 = y1 * scale + offset_y
                    screen_x2 = x2 * scale + offset_x
                    screen_y2 = y2 * scale + offset_y
                    painter.drawLine(screen_x1, screen_y1, screen_x2, screen_y2)
                
                elif entity.get('type') == 'CIRCLE' and 'center' in entity and 'radius' in entity:
                    x, y = entity['center']
                    radius = entity['radius']
                    screen_x = x * scale + offset_x
                    screen_y = y * scale + offset_y
                    screen_radius = radius * scale
                    painter.drawEllipse(QPoint(screen_x, screen_y), screen_radius, screen_radius)
                
                elif entity.get('type') in ('POLYLINE', 'LWPOLYLINE') and 'vertices' in entity:
                    vertices = entity['vertices']
                    if len(vertices) >= 2:
                        polygon = QPolygon()
                        for x, y in vertices:
                            screen_x = x * scale + offset_x
                            screen_y = y * scale + offset_y
                            polygon.append(QPoint(screen_x, screen_y))
                        
                        is_closed = entity.get('is_closed', False)
                        if is_closed:
                            painter.drawPolygon(polygon)
                        else:
                            painter.drawPolyline(polygon)
            
            # Draw detected components if highlight_components is True
            if highlight_components and self.detected_components:
                # Filter by component types based on checkboxes
                visible_types = []
                if self.show_blocks_check.isChecked():
                    visible_types.append('Block')
                if self.show_inverters_check.isChecked():
                    visible_types.append('Inverter')
                if self.show_combiners_check.isChecked():
                    visible_types.append('Combiner')
                if self.show_strings_check.isChecked():
                    visible_types.append('String')
                if self.show_modules_check.isChecked():
                    visible_types.append('Module')
                if self.show_trackers_check.isChecked():
                    visible_types.append('Tracker')
                
                # Draw each visible component
                for component in self.detected_components:
                    if component['type'] not in visible_types:
                        continue
                    
                    if 'position' not in component:
                        continue
                    
                    x, y = component['position']
                    screen_x = x * scale + offset_x
                    screen_y = y * scale + offset_y
                    
                    color = self._get_color_for_component_type(component['type'])
                    
                    # Check if this is the selected component
                    is_selected = (self.selected_component and 
                                  self.selected_component.get('type') == component.get('type') and 
                                  self.selected_component.get('id') == component.get('id'))
                    
                    if is_selected:
                        # Highlight selected component
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.setBrush(QColor(255, 0, 0, 100))
                        marker_size = 10
                    else:
                        # Normal component
                        painter.setPen(QPen(QColor(*color), 1))
                        painter.setBrush(QColor(*color, 100))
                        marker_size = 6
                    
                    # Draw marker based on component type
                    if component['type'] == 'Block':
                        # Draw rectangle for block
                        painter.drawRect(screen_x - marker_size*2, screen_y - marker_size*2, 
                                       marker_size*4, marker_size*4)
                    elif component['type'] == 'Inverter':
                        # Draw diamond for inverter
                        diamond = QPolygon()
                        diamond.append(QPoint(screen_x, screen_y - marker_size))
                        diamond.append(QPoint(screen_x + marker_size, screen_y))
                        diamond.append(QPoint(screen_x, screen_y + marker_size))
                        diamond.append(QPoint(screen_x - marker_size, screen_y))
                        painter.drawPolygon(diamond)
                    elif component['type'] == 'Combiner':
                        # Draw circle for combiner
                        painter.drawEllipse(QPoint(screen_x, screen_y), marker_size, marker_size)
                    elif component['type'] == 'String':
                        # Draw line for string
                        painter.drawLine(screen_x - marker_size, screen_y, screen_x + marker_size, screen_y)
                        painter.drawLine(screen_x, screen_y - marker_size, screen_x, screen_y + marker_size)
                    elif component['type'] == 'Module':
                        # Draw small square for module
                        painter.drawRect(screen_x - marker_size/2, screen_y - marker_size/2, 
                                       marker_size, marker_size)
                    elif component['type'] == 'Tracker':
                        # Draw triangle for tracker
                        triangle = QPolygon()
                        triangle.append(QPoint(screen_x, screen_y - marker_size))
                        triangle.append(QPoint(screen_x + marker_size, screen_y + marker_size))
                        triangle.append(QPoint(screen_x - marker_size, screen_y + marker_size))
                        painter.drawPolygon(triangle)
                    
                    # Draw label
                    if is_selected or component['type'] in ['Block', 'Inverter', 'Combiner']:
                        painter.setPen(QPen(QColor(0, 0, 0), 1))
                        label = f"{component['type']} {component['id']}"
                        painter.drawText(QPoint(screen_x + marker_size + 2, screen_y), label)
            
            # End painter
            painter.end()
            
            # Display the image
            pixmap = QPixmap.fromImage(image)
            self.cad_view.setPixmap(pixmap)
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error visualizing CAD: {str(e)}\n{error_details}")
            self.cad_view.setText(f"Visualization Error: {str(e)}")
    
    def _get_color_for_entity_type(self, entity_type):
        """Get color for entity type."""
        # Default colors for different entity types
        color_map = {
            'TEXT': (0, 0, 255),     # Blue
            'MTEXT': (0, 0, 200),    # Dark Blue
            'LINE': (0, 0, 0),       # Black
            'CIRCLE': (255, 0, 0),   # Red
            'POLYLINE': (0, 128, 0), # Green
            'LWPOLYLINE': (0, 128, 0), # Green
            'INSERT': (128, 0, 128), # Purple
            'HATCH': (255, 128, 0),  # Orange
            'unknown': (128, 128, 128)  # Gray
        }
        
        return color_map.get(entity_type, color_map['unknown'])
    
    def _get_color_for_layer(self, layer_name):
        """Get color for layer."""
        # Hash the layer name to get a consistent color
        import hashlib
        hash_value = int(hashlib.md5(layer_name.encode()).hexdigest(), 16)
        
        r = (hash_value & 0xFF0000) >> 16
        g = (hash_value & 0x00FF00) >> 8
        b = hash_value & 0x0000FF
        
        return (r, g, b)
    
    def _get_color_for_component_type(self, component_type):
        """Get color for component type."""
        # Colors for different component types
        component_colors = {
            'block': (204, 0, 0),     # Dark Red
            'Block': (204, 0, 0),     # Dark Red
            'inverter': (0, 102, 204), # Blue
            'Inverter': (0, 102, 204), # Blue
            'combiner': (0, 153, 0),   # Green
            'Combiner': (0, 153, 0),   # Green
            'string': (153, 51, 255),  # Purple
            'String': (153, 51, 255),  # Purple
            'module': (255, 153, 0),   # Orange
            'Module': (255, 153, 0),   # Orange
            'tracker': (102, 0, 102),  # Dark Purple
            'Tracker': (102, 0, 102)   # Dark Purple
        }
        
        return component_colors.get(component_type, (128, 128, 128))  # Gray default
    
    def on_visualization_option_changed(self):
        """Handle visualization option changes."""
        self.visualize_cad(highlight_components=True)
    
    def on_zoom_in(self):
        """Zoom in on visualization."""
        self.visualization_scale *= 1.2
        self.visualize_cad(highlight_components=True)
    
    def on_zoom_out(self):
        """Zoom out on visualization."""
        self.visualization_scale /= 1.2
        self.visualize_cad(highlight_components=True)
    
    def on_reset_view(self):
        """Reset visualization view."""
        self.visualization_scale = 1.0
        self.visualization_offset = (0, 0)
        self.visualize_cad(highlight_components=True)
    
    def on_export(self, format_type=None):
        """Export site structure."""
        if not self.site_structure or not self.site_structure.structure:
            QMessageBox.warning(self, "Warning", "No site structure to export")
            return
        
        if not format_type:
            # Create dialog to select format
            export_dialog = QDialog(self)
            export_dialog.setWindowTitle("Export Format")
            export_dialog.setMinimumWidth(300)
            
            layout = QVBoxLayout(export_dialog)
            
            format_group = QButtonGroup(export_dialog)
            json_radio = QRadioButton("JSON")
            format_group.addButton(json_radio, 0)
            layout.addWidget(json_radio)
            
            csv_radio = QRadioButton("CSV")
            format_group.addButton(csv_radio, 1)
            layout.addWidget(csv_radio)
            
            if EXCEL_AVAILABLE:
                excel_radio = QRadioButton("Excel")
                format_group.addButton(excel_radio, 2)
                layout.addWidget(excel_radio)
            
            # Set JSON as default
            json_radio.setChecked(True)
            
            button_layout = QHBoxLayout()
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(export_dialog.accept)
            button_layout.addWidget(ok_button)
            
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(export_dialog.reject)
            button_layout.addWidget(cancel_button)
            
            layout.addLayout(button_layout)
            
            result = export_dialog.exec_()
            
            if result == QDialog.Accepted:
                format_id = format_group.checkedId()
                if format_id == 0:
                    format_type = 'json'
                elif format_id == 1:
                    format_type = 'csv'
                elif format_id == 2:
                    format_type = 'excel'
                else:
                    return
            else:
                return
        
        # Get file path
        if format_type == 'json':
            file_filter = "JSON files (*.json);;All files (*.*)"
            default_ext = ".json"
        elif format_type == 'csv':
            file_filter = "CSV files (*.csv);;All files (*.*)"
            default_ext = ".csv"
        elif format_type == 'excel':
            file_filter = "Excel files (*.xlsx);;All files (*.*)"
            default_ext = ".xlsx"
        else:
            QMessageBox.critical(self, "Error", f"Unsupported format type: {format_type}")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Site Structure", "", file_filter
        )
        
        if not file_path:
            return
        
        # Ensure file has correct extension
        if not file_path.lower().endswith(default_ext):
            file_path += default_ext
        
        try:
            # Update site structure with current UI values
            self.site_structure.structure['name'] = self.site_name_edit.text()
            self.site_structure.structure['module_type'] = self.module_type_edit.text()
            try:
                self.site_structure.structure['module_rated_w'] = int(self.module_power_edit.text())
            except ValueError:
                pass  # Keep existing value
            
            # Export site structure
            result = self.site_structure.export_structure(format_type, file_path)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Site structure exported to {file_path}")
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error exporting site structure: {str(e)}\n{error_details}")
            QMessageBox.critical(self, "Export Error", f"Error exporting site structure: {str(e)}")
    
    def on_about(self):
        """Show about dialog."""
        about_text = """
        <h1>SolarEye CAD Structure Analyzer</h1>
        <p>Version 1.0</p>
        <p>A tool for analyzing CAD files of solar installations and extracting structural information.</p>
        <p>Copyright © 2025</p>
        <p>Features:</p>
        <ul>
            <li>Import DXF and DWG files</li>
            <li>Automatic detection of solar components</li>
            <li>Site structure extraction</li>
            <li>Visual component mapping</li>
            <li>Export to various formats</li>
        </ul>
        """
        
        QMessageBox.about(self, "About SolarEye CAD Analyzer", about_text)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Check available CAD libraries
    if not CAD_LIBRARIES:
        QMessageBox.warning(None, "Limited Functionality", 
                          "No CAD libraries found. Install ezdxf for DXF/DWG support.")
    
    # Create and show main window
    main_window = CADAnalyzerMainWindow()
    main_window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()