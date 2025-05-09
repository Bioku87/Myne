import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import cv2
import logging
import traceback
from datetime import datetime, date
import time
import tempfile
from collections import defaultdict
import threading

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QTabWidget, QGroupBox, 
                           QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
                           QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
                           QMessageBox, QProgressBar, QSplitter, QFrame, QDialog,
                           QListWidget, QRadioButton, QButtonGroup, QScrollArea,
                           QDateEdit, QFormLayout, QToolButton, QMenu, QAction,
                           QInputDialog, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDate
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QColor, QPainter, QPen

# Try importing optional packages
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Set up logging
log_filename = f"solareye_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

# Constants for classification
DEFECT_TYPES = [
    'Cell microcrack',
    'Bypass diode issue',
    'PID degradation',
    'Soiling or shading',
    'Busbar corrosion',
    'Potential hot spot',
    'Junction box issue',
    'Module degradation',
    'Edge delamination',
    'String outage',
    'Tracker issue',
    'Combiner issue',
    'Unknown anomaly'
]

# Constants for site modeling
SITE_COMPONENT_TYPES = [
    'Block',
    'Inverter',
    'Combiner',
    'String',
    'Tracker',
    'Table',
    'Module'
]


class ThermalImageCanvas(FigureCanvasQTAgg):
    """Canvas for displaying thermal imagery with anomalies."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.thermal_data = None
        self.anomalies = None
        self.colormap = 'inferno'
        self.overlay_image = None
        self.show_overlay = False
        self.overlay_alpha = 0.3
        
    def set_data(self, thermal_data, anomalies=None):
        """Set thermal data and anomalies to display."""
        self.thermal_data = thermal_data
        self.anomalies = anomalies
        self.update_plot()
        
    def set_overlay(self, overlay_image, show=True, alpha=0.3):
        """Set overlay image data."""
        self.overlay_image = overlay_image
        self.show_overlay = show
        self.overlay_alpha = alpha
        self.update_plot()
        
    def update_plot(self):
        """Update the plot with current data."""
        if self.thermal_data is None:
            return
            
        self.axes.clear()
        
        # Convert thermal data to display format if needed
        display_data = self.thermal_data
        if len(display_data.shape) == 2:
            # Apply colormap in matplotlib instead of OpenCV
            # The data will be normalized automatically
            img = self.axes.imshow(display_data, cmap=self.colormap)
            self.fig.colorbar(img, ax=self.axes, label='Temperature (°C)')
        else:
            # Display RGB image
            img = self.axes.imshow(display_data)
        
        # Apply overlay if available
        if self.show_overlay and self.overlay_image is not None:
            # Display overlay on top with transparency
            overlay = self.axes.imshow(self.overlay_image, alpha=self.overlay_alpha)
        
        # Plot anomalies if available
        if self.anomalies:
            for anomaly in self.anomalies:
                # Get position
                if 'centroid' in anomaly:
                    x, y = anomaly['centroid']
                    
                    # Set color based on severity
                    severity = anomaly.get('severity', 'low')
                    if severity == 'high':
                        color = 'red'
                    elif severity == 'medium':
                        color = 'orange'
                    else:
                        color = 'yellow'
                    
                    # Plot marker
                    self.axes.plot(x, y, marker='x', color=color, markersize=10)
                    
                    # Plot bounding box
                    if 'bbox' in anomaly:
                        x, y, w, h = anomaly['bbox']
                        rect = plt.Rectangle((x, y), w, h, fill=False, 
                                           edgecolor=color, linewidth=1)
                        self.axes.add_patch(rect)
                        
                        # Add text label with anomaly type
                        defect_type = anomaly.get('defect_type', 'Unknown')
                        self.axes.text(x, y-5, defect_type, color='white', 
                                     fontsize=8, weight='bold',
                                     bbox=dict(facecolor=color, alpha=0.7, pad=1))
        
        self.axes.set_title('Thermal Image Analysis')
        self.fig.tight_layout()
        self.draw()


class AnalysisWorker(QThread):
    """Worker thread for running thermal analysis."""
    progress_updated = pyqtSignal(int)
    analysis_complete = pyqtSignal(dict, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, thermal_data, metadata, config):
        super().__init__()
        self.thermal_data = thermal_data
        self.metadata = metadata
        self.config = config
        self.site_model = None
        
    def set_site_model(self, site_model):
        """Set site model for component identification."""
        self.site_model = site_model
        
    def run(self):
        """Process thermal data."""
        try:
            # Preprocess thermal data
            self.progress_updated.emit(20)
            processed_data = self._preprocess()
            
            # Detect anomalies
            self.progress_updated.emit(40)
            anomalies = self._detect_anomalies(processed_data)
            
            # Classify anomalies
            self.progress_updated.emit(60)
            classified_anomalies = self._classify_anomalies(anomalies, processed_data)
            
            # Identify components
            self.progress_updated.emit(80)
            self._identify_components(classified_anomalies)
            
            # Complete
            self.progress_updated.emit(100)
            self.analysis_complete.emit(processed_data, classified_anomalies)
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in analysis: {str(e)}\n{error_details}")
            self.error_occurred.emit(str(e))
    
    def _preprocess(self):
        """Preprocess thermal data."""
        # Apply calibration
        calibration_method = self.config.get('preprocessing', {}).get('calibration_method', 'linear')
        
        # Simple linear calibration
        calibrated_data = self.thermal_data.copy().astype(float)
        
        # Get ambient temperature from metadata if available
        ambient_temp = 25.0  # Default
        if self.metadata and 'ambient_temperature' in self.metadata:
            ambient_temp = float(self.metadata['ambient_temperature'])
        
        # Apply calibration factor and offset
        cal_factor = 0.1
        cal_offset = -20
        
        if self.metadata:
            if 'calibration_factor' in self.metadata:
                cal_factor = float(self.metadata['calibration_factor'])
            if 'calibration_offset' in self.metadata:
                cal_offset = float(self.metadata['calibration_offset'])
        
        calibrated_data = calibrated_data * cal_factor + cal_offset
        
        # Apply noise reduction filter
        filtered_data = cv2.GaussianBlur(calibrated_data, (5, 5), 1.0)
        
        # Calculate statistics
        stats = {
            'mean': np.mean(filtered_data),
            'std': np.std(filtered_data),
            'min': np.min(filtered_data),
            'max': np.max(filtered_data),
            'ambient': ambient_temp
        }
        
        # Return processed data
        return {
            'raw': self.thermal_data,
            'calibrated': calibrated_data,
            'filtered': filtered_data,
            'stats': stats,
            'metadata': self.metadata
        }
    
    def _detect_anomalies(self, processed_data):
        """Detect anomalies in thermal data."""
        # Get configuration
        threshold_method = self.config.get('detection', {}).get('threshold_method', 'adaptive')
        sigma_threshold = self.config.get('detection', {}).get('sigma_threshold', 3.0)
        min_delta_t = self.config.get('detection', {}).get('min_delta_t', 5.0)
        
        # Get data and stats
        temp_data = processed_data['filtered']
        stats = processed_data['stats']
        
        # Apply thresholding
        if threshold_method.lower() == 'adaptive':
            # Use sigma-based thresholding
            threshold = stats['mean'] + sigma_threshold * stats['std']
            hot_mask = temp_data > threshold
        else:
            # Use absolute threshold
            threshold = stats['mean'] + min_delta_t
            hot_mask = temp_data > threshold
        
        # Convert to uint8 for OpenCV
        hot_mask = hot_mask.astype(np.uint8)
        
        # Find contiguous regions
        num_labels, labels, stats_output, centroids = cv2.connectedComponentsWithStats(
            hot_mask, 
            connectivity=8
        )
        
        # Create anomaly list
        anomalies = []
        
        for i in range(1, num_labels):  # Skip background label 0
            # Get region properties
            x = stats_output[i, cv2.CC_STAT_LEFT]
            y = stats_output[i, cv2.CC_STAT_TOP]
            w = stats_output[i, cv2.CC_STAT_WIDTH]
            h = stats_output[i, cv2.CC_STAT_HEIGHT]
            area = stats_output[i, cv2.CC_STAT_AREA]
            
            # Skip tiny regions (likely noise)
            if area < 9:  # 3x3 pixels minimum
                continue
                
            # Get region statistics
            region_mask = labels == i
            region_temps = temp_data[region_mask]
            
            max_temp = np.max(region_temps)
            mean_temp = np.mean(region_temps)
            delta_t = max_temp - stats['mean']
            
            # Add to anomaly list
            anomaly = {
                'id': i,
                'bbox': (x, y, w, h),
                'centroid': centroids[i],
                'area': area,
                'max_temp': max_temp,
                'mean_temp': mean_temp,
                'delta_t': delta_t,
                'aspect_ratio': w / h if h > 0 else 0
            }
            
            # Calculate shape descriptors for better classification
            contours, _ = cv2.findContours(
                region_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                anomaly['perimeter'] = perimeter
                
                # Circularity (1 for perfect circle)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    anomaly['circularity'] = circularity
                
                # Convex hull measures
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    anomaly['solidity'] = solidity
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _classify_anomalies(self, anomalies, processed_data):
        """Classify anomalies by type."""
        classified_anomalies = []
        
        for anomaly in anomalies:
            # Determine defect type - rule-based classification
            defect_type = self._classify_with_rules(anomaly)
            
            # Determine severity
            delta_t = anomaly['delta_t']
            if delta_t > 15 or delta_t < -15:
                severity = 'high'
            elif delta_t > 8 or delta_t < -8:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Determine anomaly pattern
            pattern = 'Single'  # Default
            
            # Check for identical anomalies
            similar_anomalies = [a for a in anomalies if 
                               abs(a['area'] - anomaly['area']) < 10 and
                               abs(a['delta_t'] - anomaly['delta_t']) < 2 and
                               a['id'] != anomaly['id']]
            
            if len(similar_anomalies) > 2:
                pattern = 'Identical'
            elif len(similar_anomalies) > 0:
                # Check if they form a larger pattern
                centroids = [a['centroid'] for a in similar_anomalies]
                centroids.append(anomaly['centroid'])
                
                # Calculate distances
                try:
                    from scipy.spatial.distance import pdist
                    if len(centroids) > 1:
                        distances = pdist(centroids)
                        if np.std(distances) < 10:
                            pattern = 'Complex'
                except ImportError:
                    pass
            
            # Add classification to anomaly
            anomaly['defect_type'] = defect_type
            anomaly['severity'] = severity
            anomaly['pattern'] = pattern
            
            classified_anomalies.append(anomaly)
        
        return classified_anomalies
    
    def _classify_with_rules(self, anomaly):
        """Rule-based classification of anomalies."""
        # Extract relevant features
        delta_t = anomaly['delta_t']
        area = anomaly['area']
        aspect_ratio = anomaly['aspect_ratio']
    
        # Determine defect type based on features
        if aspect_ratio > 3:
            # Elongated shape
            defect_type = 'Bypass diode issue'
        elif aspect_ratio < 0.33:
            # Elongated in vertical direction
            defect_type = 'Busbar corrosion'
        elif area < 20 and delta_t > 8:
            # Small hot spot
            defect_type = 'Cell microcrack'
        elif area > 100 and delta_t < 0:
            # Large cool area
            defect_type = 'Soiling or shading'
        elif area > 200:
            # Very large area might be a tracker
            defect_type = 'Tracker issue'
        elif area > 50:
            # Large area
            defect_type = 'PID degradation'
        elif area > 30 and area < 50 and delta_t > 10:
            # Medium sized hot area
            defect_type = 'Junction box issue'
        elif area > 20 and area < 40 and delta_t > 5:
            # Medium sized warm area
            defect_type = 'Module degradation'
        elif 'circularity' in anomaly and anomaly['circularity'] > 0.8:
            # Circular shape
            defect_type = 'Potential hot spot'
        elif 'solidity' in anomaly and anomaly['solidity'] < 0.6:
            # Irregular shape
            defect_type = 'Edge delamination'
        else:
            # Default
            defect_type = 'Unknown anomaly'
        
        return defect_type

    def _identify_components(self, anomalies):
        """Identify site components for anomalies."""
        if not self.site_model:
            return
    
        for anomaly in anomalies:
            # Add component type based on defect type
            defect_type = anomaly.get('defect_type', '').lower()
        
            if 'tracker' in defect_type:
                anomaly['component_type'] = 'Tracker'
            elif 'string' in defect_type or 'combiner' in defect_type:
                anomaly['component_type'] = 'String'
            elif 'bypass' in defect_type or 'diode' in defect_type or 'junction' in defect_type:
                anomaly['component_type'] = 'Module'
            elif 'cell' in defect_type or 'microcrack' in defect_type or 'hot spot' in defect_type:
                anomaly['component_type'] = 'Module'
            elif 'pid' in defect_type or 'degradation' in defect_type:
                anomaly['component_type'] = 'Module'
            elif 'soiling' in defect_type or 'shading' in defect_type:
                anomaly['component_type'] = 'Module'
            else:
                # Default to Module
                anomaly['component_type'] = 'Module'
        
            # Calculate affected modules
            if anomaly['component_type'] == 'Module':
                anomaly['affected_modules'] = 1
            elif anomaly['component_type'] == 'String':
                # Get module count from site model if available
                if self.site_model:
                    # Try to find component in site model
                    component_id = anomaly.get('component_id')
                    if component_id:
                        component = self.site_model.find_component_by_id(component_id)
                        if component and hasattr(component, 'module_count'):
                            anomaly['affected_modules'] = component.module_count
                        else:
                            anomaly['affected_modules'] = 16  # Default value
                    else:
                        anomaly['affected_modules'] = 16  # Default value
                else:
                    anomaly['affected_modules'] = 16  # Default value
            elif anomaly['component_type'] == 'Tracker':
                # Estimate modules per tracker
                anomaly['affected_modules'] = 160  # Default value
        
            # Calculate power loss
            severity_factor = self._get_severity_factor(anomaly['severity'])
            affected_modules = anomaly.get('affected_modules', 1)
        
            # Get module power from site model if available
            module_power = 100  # Default: 100W per module
            if self.site_model and hasattr(self.site_model, 'module_type'):
                # Parse power from module type (e.g., "First Solar 120W")
                try:
                    import re
                    power_match = re.search(r'(\d+)W', self.site_model.module_type)
                    if power_match:
                        module_power = int(power_match.group(1))
                except:
                    pass
        
            anomaly['power_loss_kw'] = (affected_modules * module_power * severity_factor) / 1000
        
            # Assign component ID if not already set
            if 'component_id' not in anomaly:
                self._assign_component_id(anomaly)
    
        return anomalies

    def _get_severity_factor(self, severity):
        """Get a numerical factor based on severity."""
        if severity == 'high':
            return 0.9
        elif severity == 'medium':
            return 0.5
        else:  # low
            return 0.1

    def _assign_component_id(self, anomaly):
        """Assign a component ID based on position."""
        if 'centroid' not in anomaly:
            return
    
        # Get image dimensions
        if self.thermal_data is None:
            return
    
        image_h, image_w = self.thermal_data.shape[:2]
        x, y = anomaly['centroid']
    
        # Normalize coordinates to 0-1 range
        norm_x = x / image_w
        norm_y = y / image_h
    
        # Generate component ID
        component_type = anomaly.get('component_type', 'Module')
    
        if self.site_model and hasattr(self.site_model, 'blocks'):
            # Map to site model
            block_keys = list(self.site_model.blocks.keys())
            if block_keys:
                num_blocks = len(block_keys)
                block_idx = min(int(norm_x * num_blocks), num_blocks - 1)
                block_id = block_keys[block_idx]
            
                if component_type == 'Tracker':
                    # For trackers, use Y position
                    if hasattr(self.site_model.blocks[block_id], 'trackers'):
                        trackers = self.site_model.blocks[block_id].trackers
                        if trackers:
                            tracker_keys = list(trackers.keys())
                            num_trackers = len(tracker_keys)
                            tracker_idx = min(int(norm_y * num_trackers), num_trackers - 1)
                            tracker_id = tracker_keys[tracker_idx]
                            component_id = f"{block_id}-T{tracker_id}"
                        else:
                            component_id = f"{block_id}-T{int(norm_y * 10) + 1}"
                    else:
                        component_id = f"{block_id}-T{int(norm_y * 10) + 1}"
                else:
                    # For other component types, use grid position
                    inverter_idx = int(norm_y * 3)
                    inverter_id = f"I{inverter_idx + 1}"
                
                    combiner_idx = int(norm_x * 5) % 5
                    combiner_id = f"C{combiner_idx + 1}"
                
                    if component_type == 'String':
                        string_idx = int(norm_y * 10) % 10
                        component_id = f"{block_id}-{inverter_id}-{combiner_id}-S{string_idx + 1}"
                    else:  # Module
                        module_x = int(norm_x * 20) % 20
                        module_y = int(norm_y * 20) % 20
                        component_id = f"{block_id}-{inverter_id}-{combiner_id}-M{module_x + 1}x{module_y + 1}"
            else:
                # Simple grid-based IDs if no blocks in site model
                grid_x = int(norm_x * 10) + 1
                grid_y = int(norm_y * 10) + 1
                
                if component_type == 'Tracker':
                    component_id = f"B{grid_x}-T{grid_y}"
                elif component_type == 'String':
                    component_id = f"B{grid_x}-I{grid_y}-S{int(norm_y * 5) % 5 + 1}"
                else:  # Module
                    component_id = f"B{grid_x}-I{grid_y}-M{int(norm_x * 5) % 5 + 1}x{int(norm_y * 5) % 5 + 1}"
        else:
            # Simple grid-based IDs if no site model
            grid_x = int(norm_x * 10) + 1
            grid_y = int(norm_y * 10) + 1
        
            if component_type == 'Tracker':
                component_id = f"B{grid_x}-T{grid_y}"
            elif component_type == 'String':
                component_id = f"B{grid_x}-I{grid_y}-S{int(norm_y * 5) % 5 + 1}"
            else:  # Module
                component_id = f"B{grid_x}-I{grid_y}-M{int(norm_x * 5) % 5 + 1}x{int(norm_y * 5) % 5 + 1}"
    
        anomaly['component_id'] = component_id
    
        # Add section location (for reports)
        anomaly['section_id'] = f"S{int(norm_y * 5) + 1}"
        anomaly['position'] = 'L' if norm_x < 0.5 else 'R'


class SiteModel:
    """Class for representing the structure of a solar site."""
    
    def __init__(self):
        self.blocks = {}  # block_id -> Block
        self.name = "Unnamed Site"
        self.capacity_mw = 0
        self.module_type = "Unknown"
        self.module_count = 0
        self.location = ""
        self.commissioning_date = None
        self.metadata = {}
        
    def load_from_file(self, file_path):
        """Load site model from file."""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.json':
                # Load from JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Parse site info
                self.name = data.get('name', self.name)
                self.capacity_mw = data.get('capacity_mw', self.capacity_mw)
                self.module_type = data.get('module_type', self.module_type)
                self.module_count = data.get('module_count', self.module_count)
                self.location = data.get('location', self.location)
                self.metadata = data.get('metadata', {})
                
                # Parse blocks
                for block_data in data.get('blocks', []):
                    block = Block(block_data.get('id'))
                    block.load_from_dict(block_data)
                    self.blocks[block.id] = block
                
                return True
                
            elif file_ext == '.csv':
                # Load from CSV structure file
                self._load_from_csv(file_path)
                return True
                
            elif file_ext in ['.xlsx', '.xls']:
                # Load from Excel structure file
                if PANDAS_AVAILABLE:
                    self._load_from_excel(file_path)
                    return True
                else:
                    logger.error("Pandas required for Excel import")
                    return False
                
            else:
                # Unsupported format
                logger.error(f"Unsupported file format: {file_ext}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading site model: {str(e)}")
            return False
    
    def _load_from_csv(self, file_path):
        """Load site structure from CSV file."""
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Process each row
            for row in reader:
                component_type = row.get('component_type', '')
                
                if component_type == 'Site':
                    # Site info
                    self.name = row.get('name', self.name)
                    self.capacity_mw = float(row.get('capacity_mw', 0))
                    self.module_type = row.get('module_type', self.module_type)
                    self.module_count = int(row.get('module_count', 0))
                    self.location = row.get('location', self.location)
                    
                elif component_type == 'Block':
                    # Block definition
                    block_id = row.get('id', '')
                    if block_id not in self.blocks:
                        self.blocks[block_id] = Block(block_id)
                    
                    block = self.blocks[block_id]
                    block.name = row.get('name', block_id)
                    block.capacity_mw = float(row.get('capacity_mw', 0))
                    
                elif component_type == 'Inverter':
                    # Inverter definition
                    block_id = row.get('block_id', '')
                    inverter_id = row.get('id', '')
                    
                    if block_id in self.blocks:
                        block = self.blocks[block_id]
                        inverter = Inverter(inverter_id)
                        inverter.name = row.get('name', inverter_id)
                        inverter.capacity_mw = float(row.get('capacity_mw', 0))
                        block.inverters[inverter_id] = inverter
                
                elif component_type == 'Combiner':
                    # Combiner definition
                    block_id = row.get('block_id', '')
                    inverter_id = row.get('inverter_id', '')
                    combiner_id = row.get('id', '')
                    
                    if block_id in self.blocks and inverter_id in self.blocks[block_id].inverters:
                        inverter = self.blocks[block_id].inverters[inverter_id]
                        combiner = Combiner(combiner_id)
                        combiner.name = row.get('name', combiner_id)
                        inverter.combiners[combiner_id] = combiner
                
                elif component_type == 'String':
                    # String definition
                    block_id = row.get('block_id', '')
                    inverter_id = row.get('inverter_id', '')
                    combiner_id = row.get('combiner_id', '')
                    string_id = row.get('id', '')
                    
                    if (block_id in self.blocks and 
                        inverter_id in self.blocks[block_id].inverters and
                        combiner_id in self.blocks[block_id].inverters[inverter_id].combiners):
                        
                        combiner = self.blocks[block_id].inverters[inverter_id].combiners[combiner_id]
                        string = String(string_id)
                        string.name = row.get('name', string_id)
                        string.module_count = int(row.get('module_count', 0))
                        combiner.strings[string_id] = string
                
                elif component_type == 'Tracker':
                    # Tracker definition
                    block_id = row.get('block_id', '')
                    tracker_id = row.get('id', '')
                    
                    if block_id in self.blocks:
                        block = self.blocks[block_id]
                        if not hasattr(block, 'trackers'):
                            block.trackers = {}
                        
                        tracker = Tracker(tracker_id)
                        tracker.name = row.get('name', tracker_id)
                        block.trackers[tracker_id] = tracker
    
    def _load_from_excel(self, file_path):
        """Load site structure from Excel file."""
        try:
            import pandas as pd
            
            # Load Excel file
            xls = pd.ExcelFile(file_path)
            
            # Load site sheet
            if 'Site' in xls.sheet_names:
                site_df = pd.read_excel(xls, 'Site')
                if not site_df.empty:
                    row = site_df.iloc[0]
                    self.name = row.get('name', self.name)
                    self.capacity_mw = float(row.get('capacity_mw', 0))
                    self.module_type = row.get('module_type', self.module_type)
                    self.module_count = int(row.get('module_count', 0))
                    self.location = row.get('location', self.location)
            
            # Load blocks sheet
            if 'Blocks' in xls.sheet_names:
                blocks_df = pd.read_excel(xls, 'Blocks')
                for idx, row in blocks_df.iterrows():
                    block_id = str(row['id'])
                    block = Block(block_id)
                    block.name = row.get('name', block_id)
                    block.capacity_mw = float(row.get('capacity_mw', 0))
                    self.blocks[block_id] = block
            
            # Load inverters sheet
            if 'Inverters' in xls.sheet_names:
                inverters_df = pd.read_excel(xls, 'Inverters')
                for idx, row in inverters_df.iterrows():
                    block_id = str(row['block_id'])
                    inverter_id = str(row['id'])
                    
                    if block_id in self.blocks:
                        block = self.blocks[block_id]
                        inverter = Inverter(inverter_id)
                        inverter.name = row.get('name', inverter_id)
                        inverter.capacity_mw = float(row.get('capacity_mw', 0))
                        block.inverters[inverter_id] = inverter
            
            # Load combiners sheet
            if 'Combiners' in xls.sheet_names:
                combiners_df = pd.read_excel(xls, 'Combiners')
                for idx, row in combiners_df.iterrows():
                    block_id = str(row['block_id'])
                    inverter_id = str(row['inverter_id'])
                    combiner_id = str(row['id'])
                    
                    if block_id in self.blocks and inverter_id in self.blocks[block_id].inverters:
                        inverter = self.blocks[block_id].inverters[inverter_id]
                        combiner = Combiner(combiner_id)
                        combiner.name = row.get('name', combiner_id)
                        inverter.combiners[combiner_id] = combiner
            
            # Load strings sheet
            if 'Strings' in xls.sheet_names:
                strings_df = pd.read_excel(xls, 'Strings')
                for idx, row in strings_df.iterrows():
                    block_id = str(row['block_id'])
                    inverter_id = str(row['inverter_id'])
                    combiner_id = str(row['combiner_id'])
                    string_id = str(row['id'])
                    
                    if (block_id in self.blocks and 
                        inverter_id in self.blocks[block_id].inverters and
                        combiner_id in self.blocks[block_id].inverters[inverter_id].combiners):
                        
                        combiner = self.blocks[block_id].inverters[inverter_id].combiners[combiner_id]
                        string = String(string_id)
                        string.name = row.get('name', string_id)
                        string.module_count = int(row.get('module_count', 0))
                        combiner.strings[string_id] = string
            
            # Load trackers sheet
            if 'Trackers' in xls.sheet_names:
                trackers_df = pd.read_excel(xls, 'Trackers')
                for idx, row in trackers_df.iterrows():
                    block_id = str(row['block_id'])
                    tracker_id = str(row['id'])
                    
                    if block_id in self.blocks:
                        block = self.blocks[block_id]
                        if not hasattr(block, 'trackers'):
                            block.trackers = {}
                        
                        tracker = Tracker(tracker_id)
                        tracker.name = row.get('name', tracker_id)
                        block.trackers[tracker_id] = tracker
                        
        except ImportError:
            # pandas not available
            logger.error("pandas required for Excel import")
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
    
    def get_component_count(self, component_type):
        """Get count of components by type."""
        count = 0
        
        if component_type == 'Block':
            count = len(self.blocks)
        
        elif component_type == 'Inverter':
            for block in self.blocks.values():
                count += len(block.inverters)
        
        elif component_type == 'Combiner':
            for block in self.blocks.values():
                for inverter in block.inverters.values():
                    count += len(inverter.combiners)
        
        elif component_type == 'String':
            for block in self.blocks.values():
                for inverter in block.inverters.values():
                    for combiner in inverter.combiners.values():
                        count += len(combiner.strings)
        
        elif component_type == 'Tracker':
            for block in self.blocks.values():
                if hasattr(block, 'trackers'):
                    count += len(block.trackers)
        
        elif component_type == 'Module':
            count = self.module_count
            
        return count
    
    def find_component_by_id(self, component_id):
        """Find a component by ID in the hierarchy."""
        # Parse ID format (Block-Inverter-Combiner-String or similar)
        parts = component_id.split('-')
        
        # Start at block level
        if len(parts) > 0 and parts[0] in self.blocks:
            component = self.blocks[parts[0]]
            
            # Go to inverter level
            if len(parts) > 1 and parts[1] in component.inverters:
                component = component.inverters[parts[1]]
                
                # Go to combiner level
                if len(parts) > 2 and parts[2] in component.combiners:
                    component = component.combiners[parts[2]]
                    
                    # Go to string level
                    if len(parts) > 3 and parts[3] in component.strings:
                        component = component.strings[parts[3]]
            
            # If trackers are directly under block
            elif len(parts) > 1 and hasattr(component, 'trackers') and parts[1] in component.trackers:
                component = component.trackers[parts[1]]
            
            return component
            
        return None
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'capacity_mw': self.capacity_mw,
            'module_type': self.module_type,
            'module_count': self.module_count,
            'location': self.location,
            'metadata': self.metadata,
            'blocks': [block.to_dict() for block in self.blocks.values()]
        }


class Block:
    """Represents a block in the solar farm."""
    
    def __init__(self, id):
        self.id = id
        self.name = id
        self.capacity_mw = 0
        self.inverters = {}  # inverter_id -> Inverter
        self.trackers = {}   # tracker_id -> Tracker
    
    def load_from_dict(self, data):
        """Load from dictionary."""
        self.name = data.get('name', self.name)
        self.capacity_mw = data.get('capacity_mw', self.capacity_mw)
        
        # Load inverters
        for inverter_data in data.get('inverters', []):
            inverter = Inverter(inverter_data.get('id'))
            inverter.load_from_dict(inverter_data)
            self.inverters[inverter.id] = inverter
        
        # Load trackers
        for tracker_data in data.get('trackers', []):
            tracker = Tracker(tracker_data.get('id'))
            tracker.load_from_dict(tracker_data)
            self.trackers[tracker.id] = tracker
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'capacity_mw': self.capacity_mw,
            'inverters': [inverter.to_dict() for inverter in self.inverters.values()],
            'trackers': [tracker.to_dict() for tracker in self.trackers.values()]
        }


class Inverter:
    """Represents an inverter in the solar farm."""
    
    def __init__(self, id):
        self.id = id
        self.name = id
        self.capacity_mw = 0
        self.combiners = {}  # combiner_id -> Combiner
    
    def load_from_dict(self, data):
        """Load from dictionary."""
        self.name = data.get('name', self.name)
        self.capacity_mw = data.get('capacity_mw', self.capacity_mw)
        
        # Load combiners
        for combiner_data in data.get('combiners', []):
            combiner = Combiner(combiner_data.get('id'))
            combiner.load_from_dict(combiner_data)
            self.combiners[combiner.id] = combiner
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'capacity_mw': self.capacity_mw,
            'combiners': [combiner.to_dict() for combiner in self.combiners.values()]
        }


class Combiner:
    """Represents a combiner box in the solar farm."""
    
    def __init__(self, id):
        self.id = id
        self.name = id
        self.strings = {}  # string_id -> String
    
    def load_from_dict(self, data):
        """Load from dictionary."""
        self.name = data.get('name', self.name)
        
        # Load strings
        for string_data in data.get('strings', []):
            string = String(string_data.get('id'))
            string.load_from_dict(string_data)
            self.strings[string.id] = string
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'strings': [string.to_dict() for string in self.strings.values()]
        }


class String:
    """Represents a string of modules in the solar farm."""
    
    def __init__(self, id):
        self.id = id
        self.name = id
        self.module_count = 0
    
    def load_from_dict(self, data):
        """Load from dictionary."""
        self.name = data.get('name', self.name)
        self.module_count = data.get('module_count', self.module_count)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'module_count': self.module_count
        }


class Tracker:
    """Represents a tracker in the solar farm."""
    
    def __init__(self, id):
        self.id = id
        self.name = id
        self.tables = []  # List of tables on this tracker
    
    def load_from_dict(self, data):
        """Load from dictionary."""
        self.name = data.get('name', self.name)
        self.tables = data.get('tables', self.tables)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'tables': self.tables
        }


class SiteAnomalyTracker:
    """Class for tracking site anomalies over time."""
    
    def __init__(self, site_model=None):
        self.site_model = site_model
        self.inspections = {}  # date -> [anomalies]
        self.historical_data = {}  # component_id -> {date -> status}
    
    def add_inspection(self, date, anomalies, metadata=None):
        """Add a new inspection to the tracker."""
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()
        
        self.inspections[date] = {
            'anomalies': anomalies,
            'metadata': metadata or {}
        }
        
        # Update historical data
        for anomaly in anomalies:
            component_id = anomaly.get('component_id')
            if component_id:
                if component_id not in self.historical_data:
                    self.historical_data[component_id] = {}
                
                self.historical_data[component_id][date] = {
                    'status': 'defective',
                    'anomaly_type': anomaly.get('defect_type', 'Unknown'),
                    'severity': anomaly.get('severity', 'low')
                }
    
    def get_dates(self):
        """Get list of inspection dates."""
        return sorted(self.inspections.keys())
    
    def get_inspection(self, date):
        """Get inspection data for a specific date."""
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()
            
        return self.inspections.get(date)
    
    def get_component_history(self, component_id):
        """Get historical data for a specific component."""
        return self.historical_data.get(component_id, {})
    
    def get_recurring_issues(self, lookback_count=1):
        """Get list of components with recurring issues."""
        recurring = []
        
        dates = self.get_dates()
        if len(dates) <= lookback_count:
            return recurring
        
        latest_date = dates[-1]
        previous_date = dates[-1-lookback_count]
        
        latest_inspection = self.get_inspection(latest_date)
        previous_inspection = self.get_inspection(previous_date)
        
        if not latest_inspection or not previous_inspection:
            return recurring
        
        # Create sets of defective components
        latest_components = {a.get('component_id') for a in latest_inspection['anomalies'] 
                            if a.get('component_id')}
        previous_components = {a.get('component_id') for a in previous_inspection['anomalies'] 
                              if a.get('component_id')}
        
        # Find intersection
        recurring_components = latest_components.intersection(previous_components)
        
        # Get full anomaly data for recurring issues
        for anomaly in latest_inspection['anomalies']:
            if anomaly.get('component_id') in recurring_components:
                recurring.append(anomaly)
        
        return recurring
    
    def get_statistics(self, date=None):
        """Get statistics for a specific date or latest."""
        if date is None:
            dates = self.get_dates()
            if not dates:
                return {}
            date = dates[-1]
        
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()
        
        inspection = self.get_inspection(date)
        if not inspection:
            return {}
        
        anomalies = inspection['anomalies']
        
        # Count by type
        type_counts = defaultdict(int)
        for anomaly in anomalies:
            defect_type = anomaly.get('defect_type', 'Unknown')
            type_counts[defect_type] += 1
        
        # Count by severity
        severity_counts = defaultdict(int)
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'low')
            severity_counts[severity] += 1
        
        # Count by component type
        component_counts = defaultdict(int)
        for anomaly in anomalies:
            component_type = anomaly.get('component_type', 'Unknown')
            component_counts[component_type] += 1
        
        # Calculate estimated power loss
        power_loss = 0.0
        for anomaly in anomalies:
            power_loss += anomaly.get('power_loss_kw', 0)
        
        # Calculate percentage affected
        percent_affected = 0
        if self.site_model:
            total_modules = self.site_model.get_component_count('Module')
            if total_modules > 0:
                affected_modules = sum(anomaly.get('affected_modules', 0) for anomaly in anomalies)
                percent_affected = (affected_modules / total_modules) * 100
        
        return {
            'date': date,
            'anomaly_count': len(anomalies),
            'type_counts': dict(type_counts),
            'severity_counts': dict(severity_counts),
            'component_counts': dict(component_counts),
            'power_loss_kw': power_loss,
            'percent_affected': percent_affected
        }
    
    def compare_inspections(self, date1, date2):
        """Compare two inspections and return differences."""
        if isinstance(date1, str):
            date1 = datetime.strptime(date1, '%Y-%m-%d').date()
        if isinstance(date2, str):
            date2 = datetime.strptime(date2, '%Y-%m-%d').date()
        
        inspection1 = self.get_inspection(date1)
        inspection2 = self.get_inspection(date2)
        
        if not inspection1 or not inspection2:
            return {}
        
        stats1 = self.get_statistics(date1)
        stats2 = self.get_statistics(date2)
        
        # Calculate differences
        anomaly_diff = stats2['anomaly_count'] - stats1['anomaly_count']
        power_loss_diff = stats2['power_loss_kw'] - stats1['power_loss_kw']
        percent_diff = stats2['percent_affected'] - stats1['percent_affected']
        
        # Find new issues
        components1 = {a.get('component_id') for a in inspection1['anomalies'] if a.get('component_id')}
        components2 = {a.get('component_id') for a in inspection2['anomalies'] if a.get('component_id')}
        
        new_components = components2 - components1
        resolved_components = components1 - components2
        
        # Get full anomaly data for new issues
        new_issues = [a for a in inspection2['anomalies'] if a.get('component_id') in new_components]
        
        # Get full anomaly data for resolved issues
        resolved_issues = [a for a in inspection1['anomalies'] if a.get('component_id') in resolved_components]
        
        return {
            'date1': date1,
            'date2': date2,
            'anomaly_diff': anomaly_diff,
            'power_loss_diff': power_loss_diff,
            'percent_diff': percent_diff,
            'new_issues': new_issues,
            'resolved_issues': resolved_issues,
            'stats1': stats1,
            'stats2': stats2
        }
    
    def save_to_file(self, file_path):
        """Save tracker data to file."""
        # Convert dates to strings for serialization
        data = {
            'inspections': {date.isoformat(): inspection for date, inspection in self.inspections.items()},
            'historical_data': {}
        }
        
        # Convert dates in historical data
        for component_id, history in self.historical_data.items():
            data['historical_data'][component_id] = {
                date.isoformat(): status for date, status in history.items()
            }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path):
        """Load tracker data from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert string dates back to date objects
        self.inspections = {
            datetime.strptime(date, '%Y-%m-%d').date(): inspection 
            for date, inspection in data.get('inspections', {}).items()
        }
        
        # Convert dates in historical data
        self.historical_data = {}
        for component_id, history in data.get('historical_data', {}).items():
            self.historical_data[component_id] = {
                datetime.strptime(date, '%Y-%m-%d').date(): status 
                for date, status in history.items()
            }


class EnhancedCADOverlay:
    """Enhanced CAD overlay class with panorama support."""
    
    def __init__(self):
        self.cad_image = None
        self.cad_metadata = None
        self.transform_matrix = None
        self.registration_points = []
        self.panorama_transform_matrix = None
        self.panorama_registration_points = []
        self.panorama = None  # Added to store panorama for later use
        
    def load_cad_file(self, file_path):
        """Load CAD file from various formats."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.dxf', '.dwg']:
            # Use ODA to convert to SVG if we had it
            # For now just use a placeholder
            self.cad_image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
            self.cad_metadata = {'width': 1200, 'height': 800, 'format': file_ext[1:], 'filename': file_path}
            
            # Draw some fake panel outlines for demo
            for i in range(0, 1000, 100):
                for j in range(0, 600, 200):
                    cv2.rectangle(self.cad_image, (i+20, j+20), (i+90, j+180), (100, 100, 100), 1)
                    
            return True
        
        elif file_ext in ['.svg', '.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            # Load as image
            try:
                self.cad_image = cv2.imread(file_path)
                if self.cad_image is None:
                    return False
                
                if len(self.cad_image.shape) == 2:
                    # Convert grayscale to RGB
                    self.cad_image = cv2.cvtColor(self.cad_image, cv2.COLOR_GRAY2RGB)
                
                self.cad_metadata = {
                    'width': self.cad_image.shape[1],
                    'height': self.cad_image.shape[0],
                    'format': file_ext[1:],
                    'filename': file_path
                }
                return True
            
            except Exception as e:
                logger.error(f"Error loading CAD image: {str(e)}")
                return False
        
        else:
            # Unsupported format
            return False
    
    def register_with_thermal(self, thermal_image, registration_points=None):
        """Register CAD with thermal image."""
        if self.cad_image is None or thermal_image is None:
            return False
        
        # If registration points provided, use them
        if registration_points and len(registration_points) >= 3:
            self.registration_points = registration_points
            
            # Extract source and destination points
            src_pts = np.float32([p['cad'] for p in registration_points])
            dst_pts = np.float32([p['thermal'] for p in registration_points])
            
            # Calculate transformation matrix
            self.transform_matrix = cv2.findHomography(src_pts, dst_pts)[0]
            
            return True
        
        else:
            # Use automatic registration
            # This would use feature detection, but for now we'll just resize
            thermal_h, thermal_w = thermal_image.shape[:2]
            cad_h, cad_w = self.cad_image.shape[:2]
            
            scale_x = thermal_w / cad_w
            scale_y = thermal_h / cad_h
            
            self.transform_matrix = np.array([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ])
            
            return True
    
    def register_with_panorama(self, panorama, registration_points=None):
        """Register CAD with stitched panorama image."""
        if self.cad_image is None or panorama is None:
            return False
        
        # If registration points provided, use them
        if registration_points and len(registration_points) >= 3:
            self.panorama_registration_points = registration_points
            
            # Extract source and destination points
            src_pts = np.float32([p['cad'] for p in registration_points])
            dst_pts = np.float32([p['panorama'] for p in registration_points])
            
            # Calculate transformation matrix
            self.panorama_transform_matrix = cv2.findHomography(src_pts, dst_pts)[0]
            
            return True
        
        else:
            # Try automatic registration with feature matching
            try:
                # Convert both to grayscale for better feature matching
                if len(self.cad_image.shape) == 3:
                    cad_gray = cv2.cvtColor(self.cad_image, cv2.COLOR_BGR2GRAY)
                else:
                    cad_gray = self.cad_image
                
                if len(panorama.shape) == 3:
                    panorama_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
                else:
                    panorama_gray = panorama
                
                # Detect ORB features and compute descriptors
                orb = cv2.ORB_create(nfeatures=5000)
                kp1, des1 = orb.detectAndCompute(cad_gray, None)
                kp2, des2 = orb.detectAndCompute(panorama_gray, None)
                
                # Match descriptors
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                
                # Sort them by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Take only good matches
                good_matches = matches[:min(100, len(matches))]
                
                # Get matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                self.panorama_transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # Save registration points for future use
                for i in range(min(4, len(good_matches))):
                    m = good_matches[i]
                    self.panorama_registration_points.append({
                        'cad': kp1[m.queryIdx].pt,
                        'panorama': kp2[m.trainIdx].pt
                    })
                
                return True
                
            except Exception as e:
                logger.error(f"Automatic CAD-panorama registration failed: {str(e)}")
                
                # Fall back to simple scaling if automated registration fails
                cad_h, cad_w = self.cad_image.shape[:2]
                panorama_h, panorama_w = panorama.shape[:2]
                
                scale_x = panorama_w / cad_w
                scale_y = panorama_h / cad_h
                
                self.panorama_transform_matrix = np.array([
                    [scale_x, 0, 0],
                    [0, scale_y, 0],
                    [0, 0, 1]
                ])
                
                return True
    
    def apply_overlay(self, thermal_image, alpha=0.3):
        """Apply CAD overlay to thermal image."""
        if self.cad_image is None or self.transform_matrix is None or thermal_image is None:
            return thermal_image
        
        # Get output dimensions
        thermal_h, thermal_w = thermal_image.shape[:2]
        
        # Warp CAD image to align with thermal
        warped_cad = cv2.warpPerspective(
            self.cad_image, self.transform_matrix, (thermal_w, thermal_h)
        )
        
        # Ensure thermal image is RGB for overlay
        thermal_rgb = thermal_image
        if len(thermal_image.shape) == 2:
            thermal_rgb = cv2.applyColorMap(
                cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                cv2.COLORMAP_INFERNO
            )
        
        # Create overlay
        overlay = cv2.addWeighted(thermal_rgb, 1.0, warped_cad, alpha, 0)
        
        return overlay
    
    def apply_panorama_overlay(self, panorama, alpha=0.3):
        """Apply CAD overlay to stitched panorama image."""
        if self.cad_image is None or self.panorama_transform_matrix is None or panorama is None:
            return panorama
        
        # Get output dimensions
        panorama_h, panorama_w = panorama.shape[:2]
        
        # Warp CAD image to align with panorama
        warped_cad = cv2.warpPerspective(
            self.cad_image, self.panorama_transform_matrix, (panorama_w, panorama_h)
        )
        
        # Ensure panorama is RGB for overlay
        panorama_rgb = panorama
        if len(panorama.shape) == 2:
            panorama_rgb = cv2.applyColorMap(
                cv2.normalize(panorama, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                cv2.COLORMAP_INFERNO
            )
        
        # Create overlay
        overlay = cv2.addWeighted(panorama_rgb, 1.0, warped_cad, alpha, 0)
        
        return overlay

    def draw_anomalies_on_overlay(self, overlay, anomalies):
        """Draw anomalies on overlay image."""
        if overlay is None or anomalies is None:
            return overlay
        
        # Create copy of overlay
        result = overlay.copy()
        
        # Draw each anomaly
        for anomaly in anomalies:
            # Get position
            if 'centroid' in anomaly:
                x, y = anomaly['centroid']
                
                # Set color based on severity
                severity = anomaly.get('severity', 'low')
                if severity == 'high':
                    color = (0, 0, 255)  # Red
                elif severity == 'medium':
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow
                
                # Plot marker
                cv2.drawMarker(result, (int(x), int(y)), color, 
                              cv2.MARKER_CROSS, 20, 2)
                
                # Plot bounding box
                if 'bbox' in anomaly:
                    x, y, w, h = anomaly['bbox']
                    cv2.rectangle(result, (int(x), int(y)), (int(x+w), int(y+h)), 
                                color, 2)
                    
                    # Add text label
                    defect_type = anomaly.get('defect_type', 'Unknown')
                    cv2.putText(result, defect_type, (int(x), int(y-5)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def draw_anomalies_on_panorama(self, panorama, anomalies, component_registry=None):
        """Draw anomalies on panorama image with CAD overlay."""
        if panorama is None or not anomalies:
            return panorama
        
        # Apply CAD overlay first
        result = self.apply_panorama_overlay(panorama)
        
        # Draw each anomaly
        for anomaly in anomalies:
            # Get position - need to transform from original image to panorama
            if 'centroid' in anomaly:
                x, y = anomaly['centroid']
                
                # Transform coordinates if registry provided
                if component_registry and 'original_file' in anomaly:
                    orig_file = anomaly['original_file']
                    if orig_file in component_registry:
                        # Apply transformation from registry
                        transformed_point = cv2.perspectiveTransform(
                            np.array([[[x, y]]], dtype=np.float32),
                            component_registry[orig_file]['transform']
                        )
                        x, y = transformed_point[0][0]
                
                # Set color based on severity
                severity = anomaly.get('severity', 'low')
                if severity == 'high':
                    color = (0, 0, 255)  # Red
                elif severity == 'medium':
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow
                
                # Plot marker
                cv2.drawMarker(result, (int(x), int(y)), color, 
                              cv2.MARKER_CROSS, 20, 2)
                
                # Add text label
                defect_type = anomaly.get('defect_type', 'Unknown')
                cv2.putText(result, defect_type, (int(x), int(y)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def generate_site_map(self, thermal_data, anomalies, output_path=None):
        """Generate site map with anomalies."""
        if self.cad_image is None or thermal_data is None:
            return None
        
        # Register CAD with thermal if not already done
        if self.transform_matrix is None:
            if not self.register_with_thermal(thermal_data):
                return None
        
        # Apply overlay
        overlay = self.apply_overlay(thermal_data, alpha=0.4)
        
        # Draw anomalies
        site_map = self.draw_anomalies_on_overlay(overlay, anomalies)
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, site_map)
        
        return site_map


class ImageStitcher:
    """Class for stitching multiple thermal images into a panorama."""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        if use_gpu:
            try:
                # Try to use GPU acceleration if available
                self.stitcher.setFeaturesFinder(cv2.cuda_GpuFeaturesFinder())
                self.stitcher.setWarper(cv2.cuda_WarperGpu())
            except Exception as e:
                logger.warning(f"GPU acceleration not available: {str(e)}")
                self.use_gpu = False
        
        # Store intermediate results
        self.feature_matches = []
        self.stitch_progress = 0
        self.panorama = None
        self.stitch_map = None
        self.downsample_factor = 1  # No downsampling by default
    
    def stitch_images(self, images, callback=None, downsample_factor=None):
        """Stitch a list of images into a panorama with progress reporting."""
        if len(images) < 2:
            if len(images) == 1:
                return images[0]
            return None
        
        try:
            # Apply downsampling if requested
            processed_images = images
            if downsample_factor and downsample_factor > 1:
                processed_images = []
                for i, img in enumerate(images):
                    h, w = img.shape[:2]
                    downsampled = cv2.resize(img, (w//downsample_factor, h//downsample_factor), 
                                           interpolation=cv2.INTER_AREA)
                    processed_images.append(downsampled)
                    
                    if callback:
                        progress = (i / len(images)) * 30
                        callback(progress, f"Downsampling image {i+1}/{len(images)}")
            
            # Convert images to consistent format
            for i in range(len(processed_images)):
                # Convert grayscale to RGB
                if len(processed_images[i].shape) == 2:
                    processed_images[i] = cv2.applyColorMap(
                        cv2.normalize(processed_images[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                        cv2.COLORMAP_INFERNO
                    )
                
                # Normalize to 8-bit format
                if processed_images[i].dtype != np.uint8:
                    processed_images[i] = cv2.normalize(
                        processed_images[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )
                
                if callback:
                    progress = 30 + (i / len(processed_images)) * 20
                    callback(progress, f"Preparing image {i+1}/{len(processed_images)}")
            
            # Stitch images
            if callback:
                callback(50, "Stitching images...")
            
            status, panorama = self.stitcher.stitch(processed_images)
            
            if status == cv2.Stitcher_OK:
                self.panorama = panorama
                
                # Create stitch map for visualization
                if callback:
                    callback(80, "Creating stitch map")
                
                self.stitch_map = self._create_stitch_map(processed_images, panorama)
                
                if callback:
                    callback(100, "Stitching complete")
                
                return panorama
            else:
                if callback:
                    callback(0, f"Stitching failed with status {status}")
                return None
                
        except Exception as e:
            logger.error(f"Stitching error: {str(e)}")
            if callback:
                callback(0, f"Stitching error: {str(e)}")
            return None
    
    def _create_stitch_map(self, images, panorama):
        """Create a visualization of how images were stitched."""
        if panorama is None:
            return None
        
        # Create colored visualization
        stitch_map = panorama.copy()
        
        # Add colored borders to visualize image regions
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        # Simple visualization - in a real app, we'd track the actual transforms
        x_offset = 0
        for i, img in enumerate(images):
            color = colors[i % len(colors)]
            h, w = img.shape[:2]
            
            # Draw a colored border in an approximated region
            x1 = x_offset
            y1 = 0
            x2 = min(x1 + w, stitch_map.shape[1]-1)
            y2 = min(h, stitch_map.shape[0]-1)
            
            # Draw rectangle
            cv2.rectangle(stitch_map, (x1, y1), (x2, y2), color, 3)
            
            # Add image number
            cv2.putText(stitch_map, str(i+1), (x1+10, y1+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Update offset
            x_offset += w // 2  # Overlap approximation
        
        return stitch_map


class BatchProcessor(QThread):
    """Thread for batch processing thermal files."""
    
    progress_updated = pyqtSignal(int, int, str)
    file_complete = pyqtSignal(str, dict, list)
    stitch_progress_updated = pyqtSignal(int, str)
    batch_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self, folder_path, config, file_pattern='*.*'):
        super().__init__()
        self.folder_path = folder_path
        self.config = config
        self.file_pattern = file_pattern
        self.stop_requested = False
        
        # Stitching
        self.stitch_enabled = False
        self.stitch_result = None
        self.stitch_overlay = None
        self.stitch_map = None
        self.stitcher = None
        self.stitch_downsample_factor = 2  # Default factor
        
        # Site model
        self.site_model = None
        
        # Component registry for panorama transforms
        self.component_registry = {}
    
    def set_site_model(self, site_model):
        """Set site model for component identification."""
        self.site_model = site_model
    
    def set_stitch_images(self, enabled=True):
        """Enable or disable image stitching."""
        self.stitch_enabled = enabled
    
    def stitch_progress_callback(self, progress, message):
        """Handle stitching progress callbacks."""
        self.stitch_progress_updated.emit(progress, message)
    
    def run(self):
        try:
            # Get list of files
            if self.file_pattern == '*.*':
                extensions = ['.tif', '.jpg', '.jpeg', '.png', '.npy']
                files = []
                for ext in extensions:
                    files.extend([f for f in os.listdir(self.folder_path) if f.lower().endswith(ext)])
            else:
                import glob
                pattern = os.path.join(self.folder_path, self.file_pattern)
                files = [os.path.basename(f) for f in glob.glob(pattern)]
            
            if not files:
                self.error_occurred.emit("No files", "No matching files found in selected folder")
                return
            
            total_files = len(files)
            batch_results = []
            all_images = []  # Store images for stitching
            all_anomalies = []  # Store all anomalies for combined overlay
            
            # Process each file
            for i, filename in enumerate(files):
                if self.stop_requested:
                    break
                
                try:
                    file_path = os.path.join(self.folder_path, filename)
                    self.progress_updated.emit(i+1, total_files, filename)
                    
                    # Load file
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    if file_ext == '.npy':
                        # Load NumPy array
                        thermal_data = np.load(file_path)
                    else:
                        # Load image
                        thermal_data = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
                        
                        if thermal_data is None:
                            self.error_occurred.emit(filename, "Failed to load file")
                            continue
                    
                    # Store for stitching if enabled
                    if self.stitch_enabled:
                        if self.stitch_downsample_factor > 1:
                            # Downsample for stitching to save memory
                            h, w = thermal_data.shape[:2]
                            factor = self.stitch_downsample_factor
                            downsampled = cv2.resize(thermal_data, (w//factor, h//factor), 
                                                   interpolation=cv2.INTER_AREA)
                            all_images.append(downsampled)
                        else:
                            all_images.append(thermal_data.copy())
                    
                    # Try to load metadata
                    metadata_path = os.path.splitext(file_path)[0] + '.json'
                    metadata = None
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        except:
                            metadata = None
                    
                    # Create analyzer
                    analyzer = AnalysisWorker(thermal_data, metadata, self.config)
                    
                    # Set site model if available
                    if self.site_model:
                        analyzer.set_site_model(self.site_model)
                    
                    # Process data
                    processed_data = analyzer._preprocess()
                    anomalies = analyzer._detect_anomalies(processed_data)
                    classified_anomalies = analyzer._classify_anomalies(anomalies, processed_data)
                    
                    # Add component identification
                    analyzer._identify_components(classified_anomalies)
                    
                    # Store original file reference for panorama registration
                    for anomaly in classified_anomalies:
                        anomaly['original_file'] = filename
                        anomaly['original_image_idx'] = i
                        all_anomalies.append(anomaly)
                    
                    # Add to results
                    result = {
                        'filename': filename,
                        'path': file_path,
                        'processed_data': processed_data,
                        'anomalies': classified_anomalies
                    }
                    batch_results.append(result)
                    
                    # Emit file result
                    self.file_complete.emit(filename, processed_data, classified_anomalies)
                    
                except Exception as e:
                    self.error_occurred.emit(filename, f"Error: {str(e)}")
            
            # Perform image stitching if enabled and we have images
            if self.stitch_enabled and all_images and not self.stop_requested:
                self.progress_updated.emit(total_files, total_files, "Starting image stitching...")
                
                # Create stitcher
                if self.stitcher is None:
                    self.stitcher = ImageStitcher()
                
                # Stitch images
                self.stitch_result = self.stitcher.stitch_images(
                    all_images,
                    self.stitch_progress_callback,
                    self.stitch_downsample_factor
                )
                
                if self.stitch_result is not None:
                    # Add to batch results
                    for result in batch_results:
                        result['stitch_panorama'] = self.stitch_result
                        result['stitch_map'] = self.stitcher.stitch_map
            
            # Complete
            self.batch_complete.emit(batch_results)
            
        except Exception as e:
            self.error_occurred.emit("Batch Processing", f"Error: {str(e)}")


class BatchReportDialog(QDialog):
    """Dialog for showing batch processing results."""
    
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        
        self.setWindowTitle("Batch Processing Results")
        self.setMinimumSize(600, 400)
        
        self._create_ui()
        self._populate_results()
    
    def _create_ui(self):
        layout = QVBoxLayout(self)
        
        # Summary group
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_label = QLabel()
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_group)
        
        # Files table
        files_group = QGroupBox("Files Processed")
        files_layout = QVBoxLayout(files_group)
        
        self.files_table = QTableWidget(0, 5)
        self.files_table.setHorizontalHeaderLabels([
            "File", "Status", "Anomalies", "Types", "Power Loss (kW)"
        ])
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        files_layout.addWidget(self.files_table)
        
        layout.addWidget(files_group)
        
        # Stitching result
        self.stitch_group = QGroupBox("Stitching Results")
        stitch_layout = QVBoxLayout(self.stitch_group)
        
        self.stitch_label = QLabel("No stitching performed")
        stitch_layout.addWidget(self.stitch_label)
        
        # Placeholder for stitch image
        self.stitch_image_label = QLabel()
        self.stitch_image_label.setAlignment(Qt.AlignCenter)
        stitch_layout.addWidget(self.stitch_image_label)
        
        # Add to main layout
        layout.addWidget(self.stitch_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        export_button = QPushButton("Export Report")
        export_button.clicked.connect(self.export_report)
        button_layout.addWidget(export_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def _populate_results(self):
        if not self.results:
            self.summary_label.setText("No results available")
            return
        
        # Count files and anomalies
        total_files = len(self.results)
        total_anomalies = sum(len(result.get('anomalies', [])) for result in self.results)
        
        # Count by type
        all_anomalies = []
        for result in self.results:
            all_anomalies.extend(result.get('anomalies', []))
        
        type_counts = defaultdict(int)
        for anomaly in all_anomalies:
            defect_type = anomaly.get('defect_type', 'Unknown')
            type_counts[defect_type] += 1
        
        # Calculate total power loss
        total_power_loss = sum(
            anomaly.get('power_loss_kw', 0) 
            for result in self.results 
            for anomaly in result.get('anomalies', [])
        )
        
        # Update summary
        summary_text = (
            f"Total files processed: {total_files}\n"
            f"Total anomalies detected: {total_anomalies}\n"
            f"Estimated power loss: {total_power_loss:.2f} kW\n\n"
            f"Anomalies by type:\n"
        )
        
        for defect_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            summary_text += f"- {defect_type}: {count}\n"
        
        self.summary_label.setText(summary_text)
        
        # Update files table
        self.files_table.setRowCount(total_files)
        
        for i, result in enumerate(self.results):
            filename = os.path.basename(result.get('filename', f"File {i+1}"))
            self.files_table.setItem(i, 0, QTableWidgetItem(filename))
            
            # Status
            status = "Processed"
            if 'error' in result:
                status = f"Error: {result['error']}"
            self.files_table.setItem(i, 1, QTableWidgetItem(status))
            
            # Anomalies count
            anomalies = result.get('anomalies', [])
            self.files_table.setItem(i, 2, QTableWidgetItem(str(len(anomalies))))
            
            # Types summary
            types_in_file = set(a.get('defect_type', 'Unknown') for a in anomalies)
            types_text = ", ".join(types_in_file)
            self.files_table.setItem(i, 3, QTableWidgetItem(types_text))
            
            # Power loss
            power_loss = sum(a.get('power_loss_kw', 0) for a in anomalies)
            self.files_table.setItem(i, 4, QTableWidgetItem(f"{power_loss:.2f}"))
        
        # Check for stitching results
        has_stitching = any('stitch_panorama' in result for result in self.results)
        
        if has_stitching:
            # Find first result with panorama
            panorama = None
            
            for result in self.results:
                if 'stitch_panorama' in result:
                    panorama = result['stitch_panorama']
                    break
            
            if panorama is not None:
                # Update stitching info
                self.stitch_label.setText("Stitching successful")
                
                # Convert to QImage and display
                h, w = panorama.shape[:2]
                
                # Ensure RGB format for QImage
                if len(panorama.shape) == 2:
                    panorama_rgb = cv2.applyColorMap(
                        cv2.normalize(panorama, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                        cv2.COLORMAP_INFERNO
                    )
                else:
                    panorama_rgb = panorama
                
                # Convert BGR to RGB for Qt
                panorama_rgb = cv2.cvtColor(panorama_rgb, cv2.COLOR_BGR2RGB)
                
                qimg = QImage(panorama_rgb.data, w, h, panorama_rgb.strides[0], QImage.Format_RGB888)
                
                # Scale to fit
                pixmap = QPixmap.fromImage(qimg)
                scaled_pixmap = pixmap.scaled(
                    500, 300, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                self.stitch_image_label.setPixmap(scaled_pixmap)
            else:
                self.stitch_label.setText("Stitching failed")
        else:
            self.stitch_group.setVisible(False)
    
    def export_report(self):
        """Export results to a report file."""
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "", 
            "HTML Files (*.html);;Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Export based on file type
            if file_path.lower().endswith('.html'):
                self._export_html(file_path)
            else:
                self._export_text(file_path)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Report exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def _export_html(self, file_path):
        """Export as HTML report."""
        with open(file_path, 'w') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n<head>\n')
            f.write('<title>SolarEye Batch Processing Report</title>\n')
            f.write('<style>\n')
            f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
            f.write('h1, h2 { color: #0066cc; }\n')
            f.write('table { border-collapse: collapse; width: 100%; }\n')
            f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
            f.write('th { background-color: #f2f2f2; }\n')
            f.write('tr:nth-child(even) { background-color: #f9f9f9; }\n')
            f.write('.summary { margin-bottom: 20px; }\n')
            f.write('</style>\n')
            f.write('</head>\n<body>\n')
            
            # Title
            f.write('<h1>SolarEye Batch Processing Report</h1>\n')
            f.write(f'<p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
            
            # Summary
            f.write('<div class="summary">\n')
            f.write('<h2>Summary</h2>\n')
            
            total_files = len(self.results)
            total_anomalies = sum(len(result.get('anomalies', [])) for result in self.results)
            
            # Calculate total power loss
            total_power_loss = sum(
                anomaly.get('power_loss_kw', 0) 
                for result in self.results 
                for anomaly in result.get('anomalies', [])
            )
            
            f.write(f'<p>Total files processed: {total_files}</p>\n')
            f.write(f'<p>Total anomalies detected: {total_anomalies}</p>\n')
            f.write(f'<p>Estimated power loss: {total_power_loss:.2f} kW</p>\n')
            
            # Anomalies by type
            all_anomalies = []
            for result in self.results:
                all_anomalies.extend(result.get('anomalies', []))
            
            type_counts = defaultdict(int)
            for anomaly in all_anomalies:
                defect_type = anomaly.get('defect_type', 'Unknown')
                type_counts[defect_type] += 1
            
            f.write('<h3>Anomalies by Type</h3>\n')
            f.write('<ul>\n')
            for defect_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f'<li>{defect_type}: {count}</li>\n')
            f.write('</ul>\n')
            f.write('</div>\n')
            
            # Files table
            f.write('<h2>Files Processed</h2>\n')
            f.write('<table>\n')
            f.write('<tr><th>File</th><th>Status</th><th>Anomalies</th><th>Types</th><th>Power Loss (kW)</th></tr>\n')
            
            for result in self.results:
                filename = os.path.basename(result.get('filename', "Unknown"))
                
                # Status
                status = "Processed"
                if 'error' in result:
                    status = f"Error: {result['error']}"
                
                # Anomalies count
                anomalies = result.get('anomalies', [])
                anomaly_count = len(anomalies)
                
                # Types summary
                types_in_file = set(a.get('defect_type', 'Unknown') for a in anomalies)
                types_text = ", ".join(types_in_file)
                
                # Power loss
                power_loss = sum(a.get('power_loss_kw', 0) for a in anomalies)
                
                f.write('<tr>\n')
                f.write(f'<td>{filename}</td>\n')
                f.write(f'<td>{status}</td>\n')
                f.write(f'<td>{anomaly_count}</td>\n')
                f.write(f'<td>{types_text}</td>\n')
                f.write(f'<td>{power_loss:.2f}</td>\n')
                f.write('</tr>\n')
            
            f.write('</table>\n')
            
            # Stitching results
            has_stitching = any('stitch_panorama' in result for result in self.results)
            
            if has_stitching:
                f.write('<h2>Stitching Results</h2>\n')
                
                # Find first result with panorama
                panorama_found = False
                for result in self.results:
                    if 'stitch_panorama' in result:
                        panorama_found = True
                        break
                
                if panorama_found:
                    f.write('<p>Stitching successful</p>\n')
                else:
                    f.write('<p>Stitching failed</p>\n')
            
            f.write('</body>\n</html>')
    
    def _export_text(self, file_path):
        """Export as text report."""
        with open(file_path, 'w') as f:
            # Title
            f.write('SolarEye Batch Processing Report\n')
            f.write('=' * 35 + '\n')
            f.write(f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # Summary
            f.write('Summary\n')
            f.write('-' * 20 + '\n')
            
            total_files = len(self.results)
            total_anomalies = sum(len(result.get('anomalies', [])) for result in self.results)
            
            # Calculate total power loss
            total_power_loss = sum(
                anomaly.get('power_loss_kw', 0) 
                for result in self.results 
                for anomaly in result.get('anomalies', [])
            )
            
            f.write(f'Total files processed: {total_files}\n')
            f.write(f'Total anomalies detected: {total_anomalies}\n')
            f.write(f'Estimated power loss: {total_power_loss:.2f} kW\n\n')
            
            # Anomalies by type
            all_anomalies = []
            for result in self.results:
                all_anomalies.extend(result.get('anomalies', []))
            
            type_counts = defaultdict(int)
            for anomaly in all_anomalies:
                defect_type = anomaly.get('defect_type', 'Unknown')
                type_counts[defect_type] += 1
            
            f.write('Anomalies by Type\n')
            for defect_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f'- {defect_type}: {count}\n')
            f.write('\n')
            
            # Files table
            f.write('Files Processed\n')
            f.write('-' * 20 + '\n')
            f.write('File | Status | Anomalies | Types | Power Loss (kW)\n')
            f.write('-' * 80 + '\n')
            
            for result in self.results:
                filename = os.path.basename(result.get('filename', "Unknown"))
                
                # Status
                status = "Processed"
                if 'error' in result:
                    status = f"Error: {result['error']}"
                
                # Anomalies count
                anomalies = result.get('anomalies', [])
                anomaly_count = len(anomalies)
                
                # Types summary
                types_in_file = set(a.get('defect_type', 'Unknown') for a in anomalies)
                types_text = ", ".join(types_in_file)
                
                # Power loss
                power_loss = sum(a.get('power_loss_kw', 0) for a in anomalies)
                
                f.write(f'{filename} | {status} | {anomaly_count} | {types_text} | {power_loss:.2f}\n')
            
            f.write('\n')
            
            # Stitching results
            has_stitching = any('stitch_panorama' in result for result in self.results)
            
            if has_stitching:
                f.write('Stitching Results\n')
                f.write('-' * 20 + '\n')
                
                # Find first result with panorama
                panorama_found = False
                for result in self.results:
                    if 'stitch_panorama' in result:
                        panorama_found = True
                        break
                
                if panorama_found:
                    f.write('Stitching successful\n')
                else:
                    f.write('Stitching failed\n')


class SiteSettingsDialog(QDialog):
    """Dialog for configuring site settings."""
    
    def __init__(self, site_model=None, parent=None):
        super().__init__(parent)
        self.site_model = site_model or SiteModel()
        
        self.setWindowTitle("Site Configuration")
        self.setMinimumSize(500, 400)
        self._create_ui()
    
    def _create_ui(self):
        layout = QVBoxLayout(self)
        
        # Site information form
        form_group = QGroupBox("Site Information")
        form_layout = QFormLayout(form_group)
        
        self.name_edit = QLineEdit(self.site_model.name)
        form_layout.addRow("Site Name:", self.name_edit)
        
        self.capacity_spin = QDoubleSpinBox()
        self.capacity_spin.setRange(0, 1000)
        self.capacity_spin.setValue(self.site_model.capacity_mw)
        self.capacity_spin.setSuffix(" MW")
        form_layout.addRow("Capacity:", self.capacity_spin)
        
        self.module_type_edit = QLineEdit(self.site_model.module_type)
        form_layout.addRow("Module Type:", self.module_type_edit)
        
        self.module_count_spin = QSpinBox()
        self.module_count_spin.setRange(0, 1000000)
        self.module_count_spin.setValue(self.site_model.module_count)
        form_layout.addRow("Module Count:", self.module_count_spin)
        
        self.location_edit = QLineEdit(self.site_model.location)
        form_layout.addRow("Location:", self.location_edit)
        
        self.date_edit = QDateEdit(QDate.currentDate())
        if self.site_model.commissioning_date:
            date_obj = QDate.fromString(self.site_model.commissioning_date, "yyyy-MM-dd")
            self.date_edit.setDate(date_obj)
        form_layout.addRow("Commissioning Date:", self.date_edit)
        
        layout.addWidget(form_group)
        
        # Component counts
        counts_group = QGroupBox("Component Counts")
        counts_layout = QGridLayout(counts_group)
        
        row = 0
        for component_type in SITE_COMPONENT_TYPES:
            count = self.site_model.get_component_count(component_type)
            counts_layout.addWidget(QLabel(f"{component_type}s:"), row, 0)
            counts_layout.addWidget(QLabel(str(count)), row, 1)
            row += 1
        
        layout.addWidget(counts_group)
        
        # Structure actions
        structure_group = QGroupBox("Site Structure")
        structure_layout = QVBoxLayout(structure_group)
        
        import_structure_button = QPushButton("Import Site Structure")
        import_structure_button.clicked.connect(self.import_structure)
        structure_layout.addWidget(import_structure_button)
        
        layout.addWidget(structure_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def import_structure(self):
        """Import site structure from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Site Structure", "", 
            "JSON Files (*.json);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        # Load site structure
        if self.site_model.load_from_file(file_path):
            # Update UI
            self.name_edit.setText(self.site_model.name)
            self.capacity_spin.setValue(self.site_model.capacity_mw)
            self.module_type_edit.setText(self.site_model.module_type)
            self.module_count_spin.setValue(self.site_model.module_count)
            self.location_edit.setText(self.site_model.location)
            
            # Update component counts
            QMessageBox.information(self, "Import Complete", 
                                   f"Site structure imported successfully")
        else:
            QMessageBox.critical(self, "Import Error", 
                               "Failed to import site structure")
    
    def accept(self):
        """Apply changes and accept dialog."""
        try:
            # Update site model
            self.site_model.name = self.name_edit.text()
            self.site_model.capacity_mw = self.capacity_spin.value()
            self.site_model.module_type = self.module_type_edit.text()
            self.site_model.module_count = self.module_count_spin.value()
            self.site_model.location = self.location_edit.text()
        
            # Safely handle the date
            try:
                self.site_model.commissioning_date = self.date_edit.date().toString("yyyy-MM-dd")
            except Exception:
                self.site_model.commissioning_date = datetime.now().strftime("%Y-%m-%d")
            
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save site settings: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("SolarEye - Solar PV Thermal Analyzer")
        self.setMinimumSize(1200, 800)

        # Set window icon
        try:
            icon = QIcon("solareye_icon.png")  # Use the green S circuit icon
            self.setWindowIcon(icon)
        except:
            pass
        
        # Initialize data
        self.thermal_data = None
        self.metadata = None
        self.processed_data = None
        self.anomalies = []
        self.current_thermal_path = None
        self.batch_results = []
        
        # Site model
        self.site_model = SiteModel()
        
        # Anomaly tracker
        self.anomaly_tracker = SiteAnomalyTracker()
        
        # CAD overlay
        self.cad_overlay = EnhancedCADOverlay()
        
        # Configuration
        self.config = {
            'preprocessing': {
                'calibration_method': 'linear',
                'temperature_unit': 'celsius'
            },
            'detection': {
                'threshold_method': 'adaptive',
                'sigma_threshold': 3.0,
                'min_delta_t': 5.0
            },
            'classification': {
                'method': 'rule_based'
            },
            'reporting': {
                'output_dir': 'output/',
                'include_images': True,
                'include_stats': True,
                'company_name': 'SolarEye',
                'report_title': 'Solar Site Analysis'
            }
        }
        
        # Create UI
        self._create_ui()
        self._create_menus()
        
        # Set initial state
        self._update_ui_state()
        
        # Log startup
        logger.info("Main window initialized")
    
    def _create_ui(self):
        """Create the main UI layout."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        toolbar_layout = QHBoxLayout()
        
        process_button = QPushButton("Process Data")
        process_button.clicked.connect(self.on_process)
        toolbar_layout.addWidget(process_button)
        
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.on_export)
        toolbar_layout.addWidget(export_button)
        
        toolbar_layout.addStretch()
        
        # Add status label
        self.status_label = QLabel("Ready")
        toolbar_layout.addWidget(self.status_label)
        
        main_layout.addLayout(toolbar_layout)
        
        # Create splitter for main area
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls and info
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Info tab
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        # File info
        file_group = QGroupBox("File Information")
        file_layout = QVBoxLayout(file_group)
        
        self.thermal_path_label = QLabel("Thermal Image: Not loaded")
        file_layout.addWidget(self.thermal_path_label)
        
        self.metadata_label = QLabel("Metadata: Not loaded")
        file_layout.addWidget(self.metadata_label)
        
        info_layout.addWidget(file_group)
        
        # Stats group
        stats_group = QGroupBox("Image Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No data loaded")
        stats_layout.addWidget(self.stats_label)
        
        info_layout.addWidget(stats_group)
        
        # Site info group
        site_group = QGroupBox("Site Information")
        site_layout = QVBoxLayout(site_group)
        
        self.site_name_label = QLabel("Site: Not configured")
        site_layout.addWidget(self.site_name_label)
        
        self.site_capacity_label = QLabel("Capacity: 0 MW")
        site_layout.addWidget(self.site_capacity_label)
        
        self.site_module_label = QLabel("Module Type: Unknown")
        site_layout.addWidget(self.site_module_label)
        
        site_buttons_layout = QHBoxLayout()
        
        config_site_button = QPushButton("Configure Site")
        config_site_button.clicked.connect(self.on_configure_site)
        site_buttons_layout.addWidget(config_site_button)
        
        load_site_button = QPushButton("Load Site")
        load_site_button.clicked.connect(self.on_load_site)
        site_buttons_layout.addWidget(load_site_button)
        
        site_layout.addLayout(site_buttons_layout)
        
        info_layout.addWidget(site_group)
        info_layout.addStretch()
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout(detection_group)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold Method:"))
        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems(["Adaptive", "Absolute"])
        threshold_layout.addWidget(self.threshold_method_combo)
        detection_layout.addLayout(threshold_layout)
        
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma Threshold:"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 10.0)
        self.sigma_spin.setValue(3.0)
        self.sigma_spin.setSingleStep(0.1)
        sigma_layout.addWidget(self.sigma_spin)
        detection_layout.addLayout(sigma_layout)
        
        delta_layout = QHBoxLayout()
        delta_layout.addWidget(QLabel("Min Delta T (°C):"))
        self.delta_t_spin = QDoubleSpinBox()
        self.delta_t_spin.setRange(1.0, 20.0)
        self.delta_t_spin.setValue(5.0)
        self.delta_t_spin.setSingleStep(0.5)
        delta_layout.addWidget(self.delta_t_spin)
        detection_layout.addLayout(delta_layout)
        
        settings_layout.addWidget(detection_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_edit = QLineEdit("output/")
        output_dir_layout.addWidget(self.output_dir_edit)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.on_browse_output)
        output_dir_layout.addWidget(browse_button)
        output_layout.addLayout(output_dir_layout)
        
        self.generate_excel_check = QCheckBox("Generate Excel Report")
        self.generate_excel_check.setChecked(True)
        output_layout.addWidget(self.generate_excel_check)
        
        self.generate_kml_check = QCheckBox("Generate Google Maps KML")
        self.generate_kml_check.setChecked(True)
        output_layout.addWidget(self.generate_kml_check)
        
        self.generate_pdf_check = QCheckBox("Generate PDF Report")
        self.generate_pdf_check.setChecked(True)
        output_layout.addWidget(self.generate_pdf_check)
        
        settings_layout.addWidget(output_group)
        settings_layout.addStretch()
        
        # CAD Overlay tab
        cad_tab = QWidget()
        cad_layout = QVBoxLayout(cad_tab)
        
        # CAD file section
        cad_file_group = QGroupBox("CAD File")
        cad_file_layout = QVBoxLayout(cad_file_group)
        
        self.cad_path_label = QLabel("CAD File: Not loaded")
        cad_file_layout.addWidget(self.cad_path_label)
        
        cad_buttons_layout = QHBoxLayout()
        
        load_cad_button = QPushButton("Load CAD File")
        load_cad_button.clicked.connect(self.on_load_cad)
        cad_buttons_layout.addWidget(load_cad_button)
        
        register_cad_button = QPushButton("Register with Thermal")
        register_cad_button.clicked.connect(self.on_register_cad)
        cad_buttons_layout.addWidget(register_cad_button)
        
        register_panorama_button = QPushButton("Register with Panorama")
        register_panorama_button.clicked.connect(self.register_cad_with_panorama)
        cad_buttons_layout.addWidget(register_panorama_button)
        
        cad_file_layout.addLayout(cad_buttons_layout)
        
        # Overlay options
        overlay_layout = QHBoxLayout()
        overlay_layout.addWidget(QLabel("Overlay Opacity:"))
        self.overlay_opacity_spin = QDoubleSpinBox()
        self.overlay_opacity_spin.setRange(0.0, 1.0)
        self.overlay_opacity_spin.setValue(0.3)
        self.overlay_opacity_spin.setSingleStep(0.1)
        self.overlay_opacity_spin.valueChanged.connect(self.on_overlay_opacity_changed)
        overlay_layout.addWidget(self.overlay_opacity_spin)
        
        cad_file_layout.addLayout(overlay_layout)
        
        self.show_overlay_check = QCheckBox("Show Overlay")
        self.show_overlay_check.setChecked(False)
        self.show_overlay_check.stateChanged.connect(self.on_show_overlay_changed)
        cad_file_layout.addWidget(self.show_overlay_check)
        
        cad_layout.addWidget(cad_file_group)
        
        # Site Map export
        site_map_group = QGroupBox("Site Map Generation")
        site_map_layout = QVBoxLayout(site_map_group)
        
        generate_map_button = QPushButton("Generate Site Map")
        generate_map_button.clicked.connect(self.on_generate_site_map)
        site_map_layout.addWidget(generate_map_button)
        
        cad_layout.addWidget(site_map_group)
        cad_layout.addStretch()
        
        # Historical tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        # Inspections section
        inspections_group = QGroupBox("Inspection History")
        inspections_layout = QVBoxLayout(inspections_group)
        
        self.inspections_list = QListWidget()
        self.inspections_list.currentRowChanged.connect(self.on_inspection_selected)
        inspections_layout.addWidget(self.inspections_list)
        
        inspection_buttons_layout = QHBoxLayout()
        
        add_inspection_button = QPushButton("Add Current")
        add_inspection_button.clicked.connect(self.on_add_inspection)
        inspection_buttons_layout.addWidget(add_inspection_button)
        
        compare_button = QPushButton("Compare Selected")
        compare_button.clicked.connect(self.on_compare_inspections)
        inspection_buttons_layout.addWidget(compare_button)
        
        inspections_layout.addLayout(inspection_buttons_layout)
        
        history_layout.addWidget(inspections_group)
        
        # Comparison section
        comparison_group = QGroupBox("Comparison Results")
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        comparison_layout.addWidget(self.comparison_text)
        
        history_layout.addWidget(comparison_group)
        
        # Add tabs
        tabs.addTab(info_tab, "Info")
        tabs.addTab(settings_tab, "Settings")
        tabs.addTab(cad_tab, "CAD Overlay")
        tabs.addTab(history_tab, "History")
        left_layout.addWidget(tabs)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        
        # Right panel - Image view and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Image view
        self.image_canvas = ThermalImageCanvas(right_panel)
        right_layout.addWidget(self.image_canvas)
        
        # Results table
        results_group = QGroupBox("Detected Anomalies")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels([
            "ID", "Type", "Severity", "Delta T (°C)", "Area", "Power Loss (kW)", "Location"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        right_layout.addWidget(results_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([400, 800])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
    
    def _create_menus(self):
        """Create application menus."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        import_action = file_menu.addAction("&Import Thermal Data...")
        import_action.triggered.connect(self.on_import)
        
        import_batch_action = file_menu.addAction("Import &Batch Folder...")
        import_batch_action.triggered.connect(self.on_import_batch)
        
        file_menu.addSeparator()
        
        load_cad_action = file_menu.addAction("Load CAD &File...")
        load_cad_action.triggered.connect(self.on_load_cad)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)
        
        # Site menu
        site_menu = menu_bar.addMenu("&Site")
        
        configure_site_action = site_menu.addAction("&Configure Site...")
        configure_site_action.triggered.connect(self.on_configure_site)
        
        load_site_action = site_menu.addAction("&Load Site Structure...")
        load_site_action.triggered.connect(self.on_load_site)
        
        save_site_action = site_menu.addAction("&Save Site Structure...")
        save_site_action.triggered.connect(self.on_save_site)
        
        site_menu.addSeparator()
        
        load_history_action = site_menu.addAction("Load Inspection &History...")
        load_history_action.triggered.connect(self.on_load_history)
        
        save_history_action = site_menu.addAction("Save Inspection &History...")
        save_history_action.triggered.connect(self.on_save_history)
        
        # Analysis menu
        analysis_menu = menu_bar.addMenu("&Analysis")
        
        process_action = analysis_menu.addAction("&Process Data")
        process_action.triggered.connect(self.on_process)
        
        analysis_menu.addSeparator()
        
        generate_map_action = analysis_menu.addAction("Generate Site &Map...")
        generate_map_action.triggered.connect(self.on_generate_site_map)
        
        # Export menu
        export_menu = menu_bar.addMenu("&Export")
        
        excel_action = export_menu.addAction("Export to &Excel...")
        excel_action.triggered.connect(self.on_export_excel)
        
        kml_action = export_menu.addAction("Export to &Google Maps KML...")
        kml_action.triggered.connect(self.on_export_kml)
        
        pdf_action = export_menu.addAction("Generate &PDF Report...")
        pdf_action.triggered.connect(self.on_export_pdf)
        
        health_tracker_action = export_menu.addAction("Export DC &Health Tracker...")
        health_tracker_action.triggered.connect(self.on_export_health_tracker)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self.on_about)
    
    def _update_ui_state(self):
        """Update UI state based on loaded data."""
        has_data = self.thermal_data is not None
        has_results = len(self.anomalies) > 0
        has_cad = self.cad_overlay.cad_image is not None
        
        # Update information labels
        if has_data:
            self.thermal_path_label.setText(f"Thermal Image: {os.path.basename(self.current_thermal_path)}")
            
            if self.metadata:
                metadata_str = "Metadata: Loaded"
                if 'capture_date' in self.metadata:
                    metadata_str += f" - Date: {self.metadata['capture_date']}"
                self.metadata_label.setText(metadata_str)

        if self.site_model:
            self.site_name_label.setText(f"Site: {self.site_model.name}")
            self.site_capacity_label.setText(f"Capacity: {self.site_model.capacity_mw} MW")
            self.site_module_label.setText(f"Module Type: {self.site_model.module_type}")
        
        if self.processed_data and 'stats' in self.processed_data:
            stats = self.processed_data['stats']
            stats_text = (
                f"Mean Temperature: {stats['mean']:.1f}°C\n"
                f"Min Temperature: {stats['min']:.1f}°C\n"
                f"Max Temperature: {stats['max']:.1f}°C\n"
                f"Standard Deviation: {stats['std']:.1f}°C\n"
                f"Ambient Temperature: {stats['ambient']:.1f}°C"
            )
            self.stats_label.setText(stats_text)
        else:
            self.stats_label.setText("No data loaded")
        
        if has_cad:
            self.cad_path_label.setText(f"CAD File: {self.cad_overlay.cad_metadata['filename']}")
        else:
            self.cad_path_label.setText("CAD File: Not loaded")
        
        # Update results table
        self.results_table.setRowCount(len(self.anomalies))
        
        for i, anomaly in enumerate(self.anomalies):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(anomaly['id'])))
            self.results_table.setItem(i, 1, QTableWidgetItem(anomaly['defect_type']))
            self.results_table.setItem(i, 2, QTableWidgetItem(anomaly['severity'].capitalize()))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{anomaly['delta_t']:.1f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(anomaly['area'])))
            
            # Power loss if available
            if 'power_loss_kw' in anomaly:
                self.results_table.setItem(i, 5, QTableWidgetItem(f"{anomaly['power_loss_kw']:.2f}"))
            else:
                self.results_table.setItem(i, 5, QTableWidgetItem("N/A"))
            
            # Location if available
            if 'component_id' in anomaly:
                self.results_table.setItem(i, 6, QTableWidgetItem(anomaly['component_id']))
            elif 'centroid' in anomaly:
                x, y = anomaly['centroid']
                self.results_table.setItem(i, 6, QTableWidgetItem(f"({int(x)}, {int(y)})"))
            else:
                self.results_table.setItem(i, 6, QTableWidgetItem("Unknown"))
        
        # Update inspection list
        self.inspections_list.clear()
        for date in sorted(self.anomaly_tracker.get_dates(), reverse=True):
            date_str = date.strftime("%Y-%m-%d")
            inspection = self.anomaly_tracker.get_inspection(date)
            anomaly_count = len(inspection['anomalies'])
            self.inspections_list.addItem(f"{date_str} ({anomaly_count} anomalies)")
    
    def on_import(self):
        """Import thermal data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Thermal Data", "", 
            "Thermal Files (*.tif *.tiff *.jpg *.jpeg *.png *.npy);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            self.status_label.setText(f"Loading {os.path.basename(file_path)}...")
            QApplication.processEvents()
            
            # Load file based on extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.npy':
                # Load NumPy array
                self.thermal_data = np.load(file_path)
            else:
                # Load image
                self.thermal_data = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
                
                if self.thermal_data is None:
                    QMessageBox.critical(self, "Import Error", "Failed to load thermal data")
                    return
            
            # Try to load metadata
            metadata_path = os.path.splitext(file_path)[0] + '.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                except:
                    self.metadata = None
            else:
                self.metadata = None
            
            # Save current path
            self.current_thermal_path = file_path
            
            # Reset processed data and anomalies
            self.processed_data = None
            self.anomalies = []
            
            # Update display
            self.image_canvas.set_data(self.thermal_data)
            
            # Update UI
            self.status_label.setText(f"Loaded {os.path.basename(file_path)}")
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error loading file: {str(e)}")
            self.status_label.setText("Ready")
    
    def on_import_batch(self):
        """Import batch of thermal files."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with Thermal Images"
        )
        
        if not folder_path:
            return
        
        # Ask if stitching should be performed
        stitch_reply = QMessageBox.question(
            self, "Image Stitching",
            "Would you like to attempt stitching the thermal images into a panorama?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        stitch_enabled = stitch_reply == QMessageBox.Yes
        
        # Ask for downsampling factor for large images
        downsample_factor = 1
        if stitch_enabled:
            downsample_response, ok = QInputDialog.getInt(
                self, "Downsampling Factor", 
                "Enter downsampling factor for large images (1-4, higher = lower resolution but faster):",
                2, 1, 4, 1
            )
            if ok:
                downsample_factor = downsample_response
        
        # Create batch processor
        self.batch_processor = BatchProcessor(folder_path, self.config)
        
        # Set site model
        self.batch_processor.set_site_model(self.site_model)
        
        # Enable stitching if requested
        if stitch_enabled:
            self.batch_processor.set_stitch_images(True)
            self.batch_processor.stitch_downsample_factor = downsample_factor
        
        # Connect signals
        self.batch_processor.progress_updated.connect(self.on_batch_progress)
        self.batch_processor.file_complete.connect(self.on_batch_file_complete)
        self.batch_processor.batch_complete.connect(self.on_batch_complete)
        self.batch_processor.error_occurred.connect(self.on_batch_error)
        
        if stitch_enabled:
            self.batch_processor.stitch_progress_updated.connect(self.on_stitch_progress)
        
        # Update UI
        self.status_label.setText("Starting batch processing...")
        self.progress_bar.setValue(0)
        
        # Start processing
        self.batch_processor.start()
    
    def on_batch_progress(self, current, total, filename):
        """Handle batch processing progress updates."""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Processing {current}/{total}: {filename}")
        QApplication.processEvents()
    
    def on_batch_file_complete(self, filename, processed_data, anomalies):
        """Handle completed file in batch."""
        # If this is the first file processed, display it
        if not self.thermal_data:
            # Get raw thermal data from processed data
            self.thermal_data = processed_data.get('raw')
            self.metadata = processed_data.get('metadata')
            self.processed_data = processed_data
            self.anomalies = anomalies
            self.current_thermal_path = filename
            
            # Update display
            self.image_canvas.set_data(self.thermal_data, self.anomalies)
            
            # Update UI
            self._update_ui_state()
    
    def on_stitch_progress(self, progress, message):
        """Handle stitching progress updates."""
        self.status_label.setText(f"Stitching: {message} ({progress}%)")
        QApplication.processEvents()
    
    def on_batch_complete(self, results):
        """Handle completed batch processing."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Batch processing complete")
        
        # Store batch results
        self.batch_results = results
        
        # Check if we have stitching results
        has_stitching = any('stitch_panorama' in result for result in results)
        
        if has_stitching:
            # Find the first result with panorama
            for result in results:
                if 'stitch_panorama' in result:
                    panorama = result['stitch_panorama']
                    
                    # Display the panorama
                    self.image_canvas.set_data(panorama)
                    break
        
        # Show batch report dialog
        report_dialog = BatchReportDialog(results, self)
        report_dialog.exec_()
    
    def on_batch_error(self, filename, error_msg):
        """Handle batch processing error."""
        QMessageBox.critical(self, "Processing Error", f"Error processing {filename}: {error_msg}")
    
    def on_process(self):
        """Process thermal data."""
        if self.thermal_data is None:
            QMessageBox.warning(self, "Warning", "No thermal data loaded")
            return
        
        try:
            # Update UI
            self.status_label.setText("Processing data...")
            self.progress_bar.setValue(0)
            QApplication.processEvents()
            
            # Create and start worker thread
            self.analyzer = AnalysisWorker(self.thermal_data, self.metadata, self.config)
            
            # Set site model if available
            if self.site_model:
                self.analyzer.set_site_model(self.site_model)
            
            # Connect signals
            self.analyzer.progress_updated.connect(self.progress_bar.setValue)
            self.analyzer.analysis_complete.connect(self.on_analysis_complete)
            self.analyzer.error_occurred.connect(self.on_analysis_error)
            
            # Start analysis
            self.analyzer.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Error: {str(e)}")
            self.status_label.setText("Ready")
            self.progress_bar.setValue(0)
    
    def on_analysis_complete(self, processed_data, anomalies):
        """Handle completed analysis."""
        # Store results
        self.processed_data = processed_data
        self.anomalies = anomalies
        
        # Update display
        filtered_data = processed_data.get('filtered', self.thermal_data)
        self.image_canvas.set_data(filtered_data, anomalies)
        
        # Update UI
        self.status_label.setText("Processing complete")
        self._update_ui_state()
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error."""
        QMessageBox.critical(self, "Analysis Error", f"Error: {error_msg}")
        self.status_label.setText("Ready")
        self.progress_bar.setValue(0)
    
    def on_export(self):
        """Export analysis results."""
        if not self.processed_data or not self.anomalies:
            QMessageBox.warning(self, "Warning", "No analysis results to export")
            return
        
        # Create submenu for export options
        export_menu = QMenu(self)
        
        excel_action = export_menu.addAction("Export to Excel")
        excel_action.triggered.connect(self.on_export_excel)
        
        kml_action = export_menu.addAction("Export to Google Maps KML")
        kml_action.triggered.connect(self.on_export_kml)
        
        pdf_action = export_menu.addAction("Generate PDF Report")
        pdf_action.triggered.connect(self.on_export_pdf)
        
        health_tracker_action = export_menu.addAction("Export DC Health Tracker")
        health_tracker_action.triggered.connect(self.on_export_health_tracker)
        
        # Show menu at button position
        button = self.sender()
        if button:
            export_menu.exec_(button.mapToGlobal(button.rect().bottomLeft()))
    
    def on_export_excel(self):
        """Export results to Excel file."""
        if not self.anomalies:
            QMessageBox.warning(self, "Warning", "No anomalies to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "", 
            "Excel Files (*.xlsx);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            if not EXCEL_AVAILABLE:
                QMessageBox.critical(self, "Export Error", 
                                   "Excel export requires xlsxwriter package")
                return
                
            # Create workbook
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
            
            # Create anomalies sheet
            sheet = workbook.add_worksheet("Anomalies")
            
            # Add header
            headers = [
                "ID", "Type", "Severity", "Delta T (°C)", "Area", 
                "Max Temp (°C)", "Mean Temp (°C)", "Power Loss (kW)", 
                "Component Type", "Component ID", "X", "Y"
            ]
            
            for col, header in enumerate(headers):
                sheet.write(0, col, header, header_format)
            
            # Add data
            for row, anomaly in enumerate(self.anomalies, 1):
                sheet.write(row, 0, anomaly.get('id', ''), cell_format)
                sheet.write(row, 1, anomaly.get('defect_type', 'Unknown'), cell_format)
                sheet.write(row, 2, anomaly.get('severity', 'low').capitalize(), cell_format)
                sheet.write(row, 3, anomaly.get('delta_t', 0), cell_format)
                sheet.write(row, 4, anomaly.get('area', 0), cell_format)
                sheet.write(row, 5, anomaly.get('max_temp', 0), cell_format)
                sheet.write(row, 6, anomaly.get('mean_temp', 0), cell_format)
                sheet.write(row, 7, anomaly.get('power_loss_kw', 0), cell_format)
                sheet.write(row, 8, anomaly.get('component_type', 'Unknown'), cell_format)
                sheet.write(row, 9, anomaly.get('component_id', ''), cell_format)
                
                # Position
                if 'centroid' in anomaly:
                    x, y = anomaly['centroid']
                    sheet.write(row, 10, x, cell_format)
                    sheet.write(row, 11, y, cell_format)
                else:
                    sheet.write(row, 10, 0, cell_format)
                    sheet.write(row, 11, 0, cell_format)
            
            # Add statistics sheet
            stats_sheet = workbook.add_worksheet("Statistics")
            
            # Add image stats
            stats_sheet.write(0, 0, "Image Statistics", header_format)
            
            if 'stats' in self.processed_data:
                stats = self.processed_data['stats']
                
                stats_sheet.write(1, 0, "Mean Temperature (°C)", cell_format)
                stats_sheet.write(1, 1, stats.get('mean', 0), cell_format)
                
                stats_sheet.write(2, 0, "Min Temperature (°C)", cell_format)
                stats_sheet.write(2, 1, stats.get('min', 0), cell_format)
                
                stats_sheet.write(3, 0, "Max Temperature (°C)", cell_format)
                stats_sheet.write(3, 1, stats.get('max', 0), cell_format)
                
                stats_sheet.write(4, 0, "Standard Deviation (°C)", cell_format)
                stats_sheet.write(4, 1, stats.get('std', 0), cell_format)
                
                stats_sheet.write(5, 0, "Ambient Temperature (°C)", cell_format)
                stats_sheet.write(5, 1, stats.get('ambient', 0), cell_format)
            
            # Add anomaly summary
            stats_sheet.write(7, 0, "Anomaly Summary", header_format)
            
            stats_sheet.write(8, 0, "Total Anomalies", cell_format)
            stats_sheet.write(8, 1, len(self.anomalies), cell_format)
            
            # Count by type
            type_counts = defaultdict(int)
            for anomaly in self.anomalies:
                defect_type = anomaly.get('defect_type', 'Unknown')
                type_counts[defect_type] += 1
            
            row = 10
            stats_sheet.write(row, 0, "Anomalies by Type", header_format)
            stats_sheet.write(row, 1, "Count", header_format)
            
            row += 1
            for defect_type, count in type_counts.items():
                stats_sheet.write(row, 0, defect_type, cell_format)
                stats_sheet.write(row, 1, count, cell_format)
                row += 1
            
            # Count by severity
            severity_counts = defaultdict(int)
            for anomaly in self.anomalies:
                severity = anomaly.get('severity', 'low')
                severity_counts[severity] += 1
            
            row += 1
            stats_sheet.write(row, 0, "Anomalies by Severity", header_format)
            stats_sheet.write(row, 1, "Count", header_format)
            
            row += 1
            for severity, count in severity_counts.items():
                stats_sheet.write(row, 0, severity.capitalize(), cell_format)
                stats_sheet.write(row, 1, count, cell_format)
                row += 1
            
            # Close workbook
            workbook.close()
            
            QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def on_export_kml(self):
        """Export results to KML for Google Maps."""
        if not self.anomalies:
            QMessageBox.warning(self, "Warning", "No anomalies to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to KML", "", 
            "KML Files (*.kml);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Ask for site coordinates
            lat, ok = QInputDialog.getDouble(
                self, "Site Latitude", "Enter site latitude:", 
                37.7749, -90, 90, 6
            )
            
            if not ok:
                return
            
            lng, ok = QInputDialog.getDouble(
                self, "Site Longitude", "Enter site longitude:", 
                -122.4194, -180, 180, 6
            )
            
            if not ok:
                return
            
            # Create KML file
            with open(file_path, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
                f.write('<Document>\n')
                f.write(f'  <name>{self.site_model.name} Thermal Analysis</name>\n')
                f.write('  <description>Thermal anomalies detected in solar farm</description>\n')
                
                # Add styles for different severity levels
                f.write('  <Style id="highSeverity">\n')
                f.write('    <IconStyle>\n')
                f.write('      <color>ff0000ff</color>\n')
                f.write('      <scale>1.2</scale>\n')
                f.write('      <Icon>\n')
                f.write('        <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png</href>\n')
                f.write('      </Icon>\n')
                f.write('    </IconStyle>\n')
                f.write('  </Style>\n')
                
                f.write('  <Style id="mediumSeverity">\n')
                f.write('    <IconStyle>\n')
                f.write('      <color>ff00a5ff</color>\n')
                f.write('      <scale>1.0</scale>\n')
                f.write('      <Icon>\n')
                f.write('        <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>\n')
                f.write('      </Icon>\n')
                f.write('    </IconStyle>\n')
                f.write('  </Style>\n')
                
                f.write('  <Style id="lowSeverity">\n')
                f.write('    <IconStyle>\n')
                f.write('      <color>ff00ffff</color>\n')
                f.write('      <scale>0.8</scale>\n')
                f.write('      <Icon>\n')
                f.write('        <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>\n')
                f.write('      </Icon>\n')
                f.write('    </IconStyle>\n')
                f.write('  </Style>\n')
                
                # Calculate image dimensions
                img_h, img_w = self.thermal_data.shape[:2]
                
                # Add placemarks for each anomaly
                for anomaly in self.anomalies:
                    # Get position
                    if 'centroid' not in anomaly:
                        continue
                    
                    x, y = anomaly['centroid']
                    
                    # Normalize to 0-1 range
                    norm_x = x / img_w
                    norm_y = y / img_h
                    
                    # Convert to a small offset from site center (approx 100m per 0.001 degree)
                    # Adjust this scale factor based on site size
                    scale = 0.005  # Controls the spread of points
                    offset_lat = (0.5 - norm_y) * scale
                    offset_lng = (norm_x - 0.5) * scale
                    
                    # Calculate coordinates
                    point_lat = lat + offset_lat
                    point_lng = lng + offset_lng
                    
                    # Get severity for styling
                    severity = anomaly.get('severity', 'low')
                    style_id = f"{severity}Severity"
                    
                    # Get info for description
                    defect_type = anomaly.get('defect_type', 'Unknown')
                    delta_t = anomaly.get('delta_t', 0)
                    power_loss = anomaly.get('power_loss_kw', 0)
                    component_id = anomaly.get('component_id', 'Unknown')
                    
                    # Write placemark
                    f.write('  <Placemark>\n')
                    f.write(f'    <name>{defect_type}</name>\n')
                    f.write('    <description><![CDATA[\n')
                    f.write(f'      <p><b>Type:</b> {defect_type}</p>\n')
                    f.write(f'      <p><b>Severity:</b> {severity.capitalize()}</p>\n')
                    f.write(f'      <p><b>Delta T:</b> {delta_t:.1f}°C</p>\n')
                    f.write(f'      <p><b>Power Loss:</b> {power_loss:.2f} kW</p>\n')
                    f.write(f'      <p><b>Component ID:</b> {component_id}</p>\n')
                    f.write('    ]]></description>\n')
                    f.write(f'    <styleUrl>#{style_id}</styleUrl>\n')
                    f.write('    <Point>\n')
                    f.write(f'      <coordinates>{point_lng},{point_lat},0</coordinates>\n')
                    f.write('    </Point>\n')
                    f.write('  </Placemark>\n')
                
                f.write('</Document>\n')
                f.write('</kml>\n')
            
            QMessageBox.information(self, "Export Complete", 
                                  f"KML exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export KML: {str(e)}")
    
    def on_export_pdf(self):
        """Generate PDF report."""
        if not self.processed_data or not self.anomalies:
            QMessageBox.warning(self, "Warning", "No analysis results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Generate PDF Report", "", 
            "PDF Files (*.pdf);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            if not PDF_AVAILABLE:
                QMessageBox.critical(self, "Export Error", 
                                   "PDF export requires fpdf package")
                return
                
            # Get screenshot of current view for report
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Get current display image
                display_data = self.thermal_data
                if self.processed_data and 'filtered' in self.processed_data:
                    display_data = self.processed_data['filtered']
                    
                # Convert to 8-bit RGB for saving
                if len(display_data.shape) == 2:
                    img = cv2.applyColorMap(
                        cv2.normalize(display_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                        cv2.COLORMAP_INFERNO
                    )
                else:
                    img = display_data.copy()
                    
                # Draw anomalies
                if self.anomalies:
                    for anomaly in self.anomalies:
                        if 'centroid' in anomaly:
                            x, y = anomaly['centroid']
                            
                            # Set color based on severity
                            severity = anomaly.get('severity', 'low')
                            if severity == 'high':
                                color = (0, 0, 255)  # Red
                            elif severity == 'medium':
                                color = (0, 165, 255)  # Orange
                            else:
                                color = (0, 255, 255)  # Yellow
                            
                            # Plot marker
                            cv2.drawMarker(img, (int(x), int(y)), color, 
                                          cv2.MARKER_CROSS, 20, 2)
                
                # Save image
                cv2.imwrite(temp_path, img)
                
            # Simple PDF report using FPDF
            class PDF(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 15)
                    self.cell(0, 10, 'SolarEye Analysis Report', 0, 1, 'C')
                    self.ln(10)
                    
                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
            pdf = PDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, self.site_model.name, 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'L')
            pdf.ln(5)
            
            # Add thermal image
            pdf.image(temp_path, x=None, y=None, w=180)
            pdf.ln(5)
            
            # Summary
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Summary', 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            
            pdf.cell(0, 10, f"Total anomalies detected: {len(self.anomalies)}", 0, 1, 'L')
            
            # Calculate estimated power loss
            power_loss = sum(a.get('power_loss_kw', 0) for a in self.anomalies)
            pdf.cell(0, 10, f"Estimated power loss: {power_loss:.2f} kW", 0, 1, 'L')
            
            # Calculate percentage affected
            if self.site_model:
                total_modules = self.site_model.get_component_count('Module')
                if total_modules > 0:
                    affected_modules = sum(a.get('affected_modules', 0) for a in self.anomalies)
                    percent_affected = (affected_modules / total_modules) * 100
                    pdf.cell(0, 10, f"Percentage of modules affected: {percent_affected:.2f}%", 0, 1, 'L')
            
            pdf.ln(5)
            
            # Anomaly table
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Detected Anomalies', 0, 1, 'L')
            
            # Define table columns
            col_widths = [15, 40, 20, 25, 25, 60]
            
            # Table header
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(col_widths[0], 10, 'ID', 1, 0, 'C')
            pdf.cell(col_widths[1], 10, 'Type', 1, 0, 'C')
            pdf.cell(col_widths[2], 10, 'Severity', 1, 0, 'C')
            pdf.cell(col_widths[3], 10, 'Delta T (°C)', 1, 0, 'C')
            pdf.cell(col_widths[4], 10, 'Power Loss (kW)', 1, 0, 'C')
            pdf.cell(col_widths[5], 10, 'Location', 1, 1, 'C')
            
            # Table data
            pdf.set_font('Arial', '', 10)
            for anomaly in self.anomalies:
                anomaly_id = str(anomaly.get('id', ''))
                defect_type = str(anomaly.get('defect_type', 'Unknown'))
                severity = str(anomaly.get('severity', 'low')).capitalize()
                delta_t = f"{anomaly.get('delta_t', 0):.1f}"
                power_loss = f"{anomaly.get('power_loss_kw', 0):.2f}"
                
                # Location
                if 'component_id' in anomaly:
                    location = anomaly['component_id']
                elif 'centroid' in anomaly:
                    x, y = anomaly['centroid']
                    location = f"({int(x)}, {int(y)})"
                else:
                    location = "Unknown"
                
                pdf.cell(col_widths[0], 10, anomaly_id, 1, 0, 'C')
                pdf.cell(col_widths[1], 10, defect_type, 1, 0, 'L')
                pdf.cell(col_widths[2], 10, severity, 1, 0, 'C')
                pdf.cell(col_widths[3], 10, delta_t, 1, 0, 'C')
                pdf.cell(col_widths[4], 10, power_loss, 1, 0, 'C')
                pdf.cell(col_widths[5], 10, location, 1, 1, 'L')
            
            # Save PDF
            pdf.output(file_path)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            QMessageBox.information(self, "Export Complete", 
                                  f"PDF report generated: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to generate PDF: {str(e)}")
    
    def on_export_health_tracker(self):
        """Export DC Health Tracker report."""
        if not self.anomalies:
            QMessageBox.warning(self, "Warning", "No anomalies to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export DC Health Tracker", "", 
            "Excel Files (*.xlsx);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            if not EXCEL_AVAILABLE:
                QMessageBox.critical(self, "Export Error", 
                                   "Excel export requires xlsxwriter package")
                return
                
            # Add current inspection to tracker if not already present
            current_date = date.today()
            
            # Check if today's inspection already exists
            if current_date not in self.anomaly_tracker.get_dates():
                self.anomaly_tracker.add_inspection(
                    current_date, 
                    self.anomalies,
                    {
                        'filename': self.current_thermal_path
                    }
                )
            
            # Create workbook
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
            
            # Create summary sheet
            summary_sheet = workbook.add_worksheet('Summary')
            
            # Add site information
            summary_sheet.write(0, 0, 'Site', header_format)
            summary_sheet.write(0, 1, self.site_model.name, cell_format)
            
            # Add test information
            summary_sheet.write(1, 0, 'Test Information', header_format)
            summary_sheet.write(1, 1, 'Information', header_format)
            summary_sheet.write(1, 2, 'Equipment', header_format)
            
            row = 2
            summary_sheet.write(row, 0, 'Test Team', cell_format)
            summary_sheet.write(row, 1, 'SolarEye', cell_format)
            row += 1
            
            summary_sheet.write(row, 0, 'Drone ID', cell_format)
            summary_sheet.write(row, 1, '2', cell_format)
            summary_sheet.write(row, 2, 'Manufacturer/Model', cell_format)
            if self.site_model.module_type:
                summary_sheet.write(row, 3, self.site_model.module_type, cell_format)
            else:
                summary_sheet.write(row, 3, 'Unknown', cell_format)
            row += 1
            
            summary_sheet.write(row, 0, 'Camera ID', cell_format)
            summary_sheet.write(row, 1, 'Flir X72 640 Fast', cell_format)
            summary_sheet.write(row, 2, 'Inverter Type', cell_format)
            summary_sheet.write(row, 3, 'N/A', cell_format)
            summary_sheet.write(row, 4, 'No. of Modules per String', cell_format)
            summary_sheet.write(row, 5, '11', cell_format)
            row += 1
            
            summary_sheet.write(row, 0, 'Test Date', cell_format)
            summary_sheet.write(row, 1, current_date.strftime('%m/%d/%Y'), cell_format)
            summary_sheet.write(row, 2, 'Module Nominal Power (W)', cell_format)
            summary_sheet.write(row, 3, '100', cell_format)
            summary_sheet.write(row, 4, 'String Nominal Power (W)', cell_format)
            summary_sheet.write(row, 5, '1800', cell_format)
            row += 2
            
            # Add findings section
            summary_sheet.write(row, 0, 'FINDINGS', header_format)
            row += 1
            
            summary_sheet.write(row, 0, 'Method', header_format)
            summary_sheet.write(row, 1, 'Total', header_format)
            summary_sheet.write(row, 2, 'Single', header_format)
            summary_sheet.write(row, 3, 'Identical', header_format)
            summary_sheet.write(row, 4, 'Complex', header_format)
            summary_sheet.write(row, 5, 'Individual Strings', header_format)
            summary_sheet.write(row, 6, 'Modules', header_format)
            summary_sheet.write(row, 7, 'Module Substrings', header_format)
            summary_sheet.write(row, 8, 'Trackers', header_format)
            summary_sheet.write(row, 9, 'Hot Spots', header_format)
            summary_sheet.write(row, 10, 'Total Modules', header_format)
            summary_sheet.write(row, 11, 'Reviewed Modules', header_format)
            row += 1
            
            # Get anomaly counts
            all_anomalies = self.anomalies
            
            # Count by component type
            string_count = len([a for a in all_anomalies if a.get('component_type') == 'String'])
            module_count = len([a for a in all_anomalies if a.get('component_type') == 'Module'])
            substring_count = len([a for a in all_anomalies if a.get('component_type') == 'Substring'])
            tracker_count = len([a for a in all_anomalies if a.get('component_type') == 'Tracker'])
            hotspot_count = len([a for a in all_anomalies if a.get('defect_type') == 'Potential hot spot'])
            
            # Count by anomaly pattern
            single_count = len([a for a in all_anomalies if a.get('pattern') == 'Single'])
            identical_count = len([a for a in all_anomalies if a.get('pattern') == 'Identical'])
            complex_count = len([a for a in all_anomalies if a.get('pattern') == 'Complex'])
            
            total_count = len(all_anomalies)
            
            # Add total modules count
            total_modules = 0
            if self.site_model:
                total_modules = self.site_model.get_component_count('Module')
            
            # Write counts
            summary_sheet.write(row, 0, 'Current', cell_format)
            summary_sheet.write(row, 1, total_count, cell_format)
            summary_sheet.write(row, 2, single_count, cell_format)
            summary_sheet.write(row, 3, identical_count, cell_format)
            summary_sheet.write(row, 4, complex_count, cell_format)
            summary_sheet.write(row, 5, string_count, cell_format)
            summary_sheet.write(row, 6, module_count, cell_format)
            summary_sheet.write(row, 7, substring_count, cell_format)
            summary_sheet.write(row, 8, tracker_count, cell_format)
            summary_sheet.write(row, 9, hotspot_count, cell_format)
            summary_sheet.write(row, 10, total_modules, cell_format)
            summary_sheet.write(row, 11, total_modules, cell_format)
            row += 1
            
            # Add analysis row
            summary_sheet.write(row, 0, 'Analysis %', cell_format)
            
            # Calculate percentages
            percent_affected = 0.0
            affected_modules = sum(a.get('affected_modules', 0) for a in all_anomalies)
            
            if total_modules > 0:
                percent_affected = (affected_modules / total_modules) * 100
            
            summary_sheet.write(row, 1, f"{percent_affected:.2f}%", cell_format)
            
            # Add other cells
            for col in range(2, 12):
                summary_sheet.write(row, col, "", cell_format)
            
            row += 2
            
            # Create anomalies sheet
            anomalies_sheet = workbook.add_worksheet('Anomalies')
            
            # Add header
            cols = [
                'Date', 'Module', 'B', 'SECTION', 'CB', 'L/R', 'Affected Array', 
                '2019_STRING_# DOWN', '2019_TRACKER_DOWN', '2020_STRING_# DOWN', '2020_TRACKER_DOWN',
                'Number of Modules', 'Priority Level', 'Defect', 'Size Reported',
                'Nominal Power Loss (W)', 'Nominal Power (W)'
            ]
            
            for i, col in enumerate(cols):
                anomalies_sheet.write(0, i, col, header_format)
            
            # Add anomaly data
            row = 1
            for anomaly_idx, anomaly in enumerate(all_anomalies):
                # Format date
                anomalies_sheet.write(row, 0, current_date.strftime('%m/%d/%Y'), cell_format)
                
                # Add component info
                block_id = anomaly.get('block_id', '')
                section_id = anomaly.get('section_id', '')
                combiner_id = anomaly.get('combiner_id', '')
                position = anomaly.get('position', '')
                component_id = anomaly.get('component_id', '')
                
                anomalies_sheet.write(row, 1, 'D', cell_format)  # Module placeholder
                anomalies_sheet.write(row, 2, block_id, cell_format)
                anomalies_sheet.write(row, 3, section_id, cell_format)
                anomalies_sheet.write(row, 4, combiner_id, cell_format)
                anomalies_sheet.write(row, 5, position, cell_format)
                anomalies_sheet.write(row, 6, component_id, cell_format)
                
                # Add flags for historical comparison
                anomalies_sheet.write(row, 7, 'FALSE', cell_format)  # 2019 string
                anomalies_sheet.write(row, 8, 'FALSE', cell_format)  # 2019 tracker
                anomalies_sheet.write(row, 9, 'FALSE', cell_format)  # 2020 string
                anomalies_sheet.write(row, 10, 'FALSE', cell_format)  # 2020 tracker
                
                # Add anomaly details
                affected_modules = anomaly.get('affected_modules', 0)
                severity = anomaly.get('severity', 'low').capitalize()
                defect_type = anomaly.get('defect_type', 'Unknown')
                area = anomaly.get('area', 0)
                power_loss_w = anomaly.get('power_loss_kw', 0) * 1000  # Convert to watts
                
                anomalies_sheet.write(row, 11, affected_modules, cell_format)
                anomalies_sheet.write(row, 12, f"{anomaly_idx+1} - Medium-Level Power Loss", cell_format)
                anomalies_sheet.write(row, 13, defect_type, cell_format)
                anomalies_sheet.write(row, 14, area, cell_format)
                anomalies_sheet.write(row, 15, power_loss_w, cell_format)
                
                # Calculate nominal power
                module_power = 100  # Default: 100W per module
                if self.site_model and hasattr(self.site_model, 'module_type'):
                    # Try to parse power from module type
                    try:
                        import re
                        power_match = re.search(r'(\d+)W', self.site_model.module_type)
                        if power_match:
                            module_power = int(power_match.group(1))
                    except:
                        pass
                
                nominal_power = affected_modules * module_power
                anomalies_sheet.write(row, 16, nominal_power, cell_format)
                
                row += 1
            
            workbook.close()
            
            QMessageBox.information(self, "Export Complete", f"DC Health Tracker exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def on_load_cad(self):
        """Load CAD file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CAD File", "", 
            "CAD Files (*.dxf *.dwg *.svg *.png *.jpg *.jpeg);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            self.status_label.setText(f"Loading CAD file: {os.path.basename(file_path)}...")
            QApplication.processEvents()
            
            # Load CAD file
            if self.cad_overlay.load_cad_file(file_path):
                self.status_label.setText(f"CAD file loaded: {os.path.basename(file_path)}")
                
                # Update UI
                self._update_ui_state()
                
                # Show CAD overlay as default
                self.show_overlay_check.setChecked(True)
                self.on_show_overlay_changed(Qt.Checked)
            else:
                QMessageBox.critical(self, "CAD Load Error", 
                                   "Failed to load CAD file")
                self.status_label.setText("Ready")
                
        except Exception as e:
            QMessageBox.critical(self, "CAD Load Error", f"Error: {str(e)}")
            self.status_label.setText("Ready")
    
    def on_register_cad(self):
        """Register CAD file with thermal image."""
        if self.cad_overlay.cad_image is None or self.thermal_data is None:
            QMessageBox.warning(self, "Warning", 
                              "Both CAD file and thermal data must be loaded")
            return
        
        try:
            self.status_label.setText("Registering CAD with thermal image...")
            QApplication.processEvents()
            
            # Register CAD with thermal
            if self.cad_overlay.register_with_thermal(self.thermal_data):
                self.status_label.setText("CAD registered with thermal image")
                
                # Update display
                self.on_show_overlay_changed(
                    Qt.Checked if self.show_overlay_check.isChecked() else Qt.Unchecked
                )
            else:
                QMessageBox.warning(self, "Registration Warning", 
                                  "CAD registration partially failed")
                self.status_label.setText("Ready")
                
        except Exception as e:
            QMessageBox.critical(self, "Registration Error", f"Error: {str(e)}")
            self.status_label.setText("Ready")
    
    def register_cad_with_panorama(self):
        """Register CAD with panorama."""
        if self.cad_overlay.cad_image is None:
            QMessageBox.warning(self, "Warning", "CAD file must be loaded")
            return False
        
        panorama = None
        
        # Check if we have a panorama from batch processing
        if hasattr(self.cad_overlay, 'panorama') and self.cad_overlay.panorama is not None:
            panorama = self.cad_overlay.panorama
        elif self.batch_results and 'stitch_panorama' in self.batch_results[0]:
            panorama = self.batch_results[0]['stitch_panorama']
            self.cad_overlay.panorama = panorama
        
        if panorama is None:
            QMessageBox.warning(self, "Warning", 
                              "No panorama available. Process batch images first.")
            return False
        
        try:
            self.status_label.setText("Registering CAD with panorama...")
            QApplication.processEvents()
            
            # Register CAD with panorama
            if self.cad_overlay.register_with_panorama(panorama):
                self.status_label.setText("CAD registered with panorama")
                
                # Create overlay with anomalies
                if self.batch_results and 'transformed_anomalies' in self.batch_results[0]:
                    all_anomalies = self.batch_results[0]['transformed_anomalies']
                    
                    # Apply overlay with anomalies
                    overlay = self.cad_overlay.draw_anomalies_on_panorama(
                        panorama,
                        all_anomalies,
                        self.batch_processor.component_registry if hasattr(self, 'batch_processor') else None
                    )
                    
                    # Update batch results
                    for result in self.batch_results:
                        result['stitch_overlay'] = overlay
                    
                    # Update display if currently showing stitched view
                    if self.image_canvas.thermal_data is not None and self.image_canvas.thermal_data.shape == panorama.shape:
                        self.image_canvas.set_data(overlay)
                
                return True
            else:
                QMessageBox.warning(self, "Registration Warning", 
                                  "CAD registration partially failed")
                self.status_label.setText("Ready")
                return False
                
        except Exception as e:
            QMessageBox.critical(self, "Registration Error", f"Error: {str(e)}")
            self.status_label.setText("Ready")
            return False
    
    def on_show_overlay_changed(self, state):
        """Handle overlay visibility change."""
        self.show_overlay = state == Qt.Checked
        
        if not self.thermal_data or not self.cad_overlay.cad_image:
            return
        
        # Apply overlay if visible
        if self.show_overlay:
            # Check if we need to apply CAD overlay or anomaly overlay
            if self.processed_data and self.anomalies:
                # Get filtered data for better visualization
                display_data = self.processed_data.get('filtered', self.thermal_data)
                
                # Apply CAD overlay
                overlay = self.cad_overlay.apply_overlay(
                    display_data, 
                    self.overlay_opacity_spin.value()
                )
                
                # Add anomalies
                overlay = self.cad_overlay.draw_anomalies_on_overlay(
                    overlay, 
                    self.anomalies
                )
                
                # Update display
                self.image_canvas.set_data(overlay)
            else:
                # Apply simple overlay
                overlay = self.cad_overlay.apply_overlay(
                    self.thermal_data, 
                    self.overlay_opacity_spin.value()
                )
                
                # Update display
                self.image_canvas.set_data(overlay)
        else:
            # Show original data
            if self.processed_data and self.anomalies:
                # Show processed data with anomalies
                display_data = self.processed_data.get('filtered', self.thermal_data)
                self.image_canvas.set_data(display_data, self.anomalies)
            else:
                # Show raw data
                self.image_canvas.set_data(self.thermal_data)
    
    def on_overlay_opacity_changed(self, value):
        """Handle overlay opacity change."""
        self.overlay_opacity_spin.setValue(value)
        
        # Update display if overlay visible
        if self.show_overlay_check.isChecked():
            self.on_show_overlay_changed(Qt.Checked)
    
    def on_generate_site_map(self):
        """Generate site map."""
        if self.thermal_data is None or self.cad_overlay.cad_image is None:
            QMessageBox.warning(self, "Warning", 
                              "Both thermal data and CAD file must be loaded")
            return
        
        try:
            self.status_label.setText("Generating site map...")
            QApplication.processEvents()
            
            # Generate site map
            site_map = self.cad_overlay.generate_site_map(
                self.thermal_data, 
                self.anomalies
            )
            
            if site_map is not None:
                # Save to file
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Site Map", "", 
                    "Image Files (*.png *.jpg *.tif);;All Files (*.*)"
                )
                
                if file_path:
                    cv2.imwrite(file_path, site_map)
                    
                    # Store site map in current inspection metadata
                    current_date = date.today()
                    
                    # Check if today's inspection already exists
                    if current_date in self.anomaly_tracker.get_dates():
                        inspection = self.anomaly_tracker.get_inspection(current_date)
                        if 'metadata' not in inspection:
                            inspection['metadata'] = {}
                        
                        inspection['metadata']['site_map'] = file_path
                    
                    QMessageBox.information(self, "Map Generated", 
                                         f"Site map saved to {file_path}")
                
                # Show site map
                self.image_canvas.set_data(site_map)
                
                self.status_label.setText("Site map generated")
            else:
                QMessageBox.warning(self, "Map Generation Failed", 
                                  "Failed to generate site map")
                self.status_label.setText("Ready")
                
        except Exception as e:
            QMessageBox.critical(self, "Map Generation Error", f"Error: {str(e)}")
            self.status_label.setText("Ready")
    
    def on_configure_site(self):
        """Configure site settings."""
        dialog = SiteSettingsDialog(self.site_model, self)
        if dialog.exec_() == QDialog.Accepted:
            # Apply updated site model
            self.site_model = dialog.site_model
            
            # Update UI
            self._update_ui_state()
    
    def on_load_site(self):
        """Load site structure from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Site Structure", "", 
            "Site Files (*.json *.csv *.xlsx);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load site structure
            if self.site_model.load_from_file(file_path):
                QMessageBox.information(self, "Load Complete", 
                                      "Site structure loaded successfully")
                
                # Update UI
                self._update_ui_state()
            else:
                QMessageBox.critical(self, "Load Error", 
                                   "Failed to load site structure")
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error: {str(e)}")
    
    def on_save_site(self):
        """Save site structure to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Site Structure", "", 
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Convert to JSON
            site_data = self.site_model.to_dict()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(site_data, f, indent=2)
            
            QMessageBox.information(self, "Save Complete", 
                                  f"Site structure saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error: {str(e)}")
    
    def on_load_history(self):
        """Load inspection history."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Inspection History", "", 
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load history from file
            self.anomaly_tracker.load_from_file(file_path)
            
            # Update UI
            self._update_ui_state()
            
            QMessageBox.information(self, "Load Complete", 
                                  "Inspection history loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load history: {str(e)}")
    
    def on_save_history(self):
        """Save inspection history."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Inspection History", "", 
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save history to file
            self.anomaly_tracker.save_to_file(file_path)
            
            QMessageBox.information(self, "Save Complete", 
                                  f"Inspection history saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save history: {str(e)}")
    
    def on_inspection_selected(self, row):
        """Handle inspection selection."""
        if row < 0:
            return
        
        dates = self.anomaly_tracker.get_dates()
        if row < len(dates):
            selected_date = dates[len(dates) - 1 - row]  # Reverse order
            inspection = self.anomaly_tracker.get_inspection(selected_date)
            
            if inspection:
                # Show inspection details
                self.comparison_text.clear()
                
                anomalies = inspection['anomalies']
                
                self.comparison_text.append(f"Inspection date: {selected_date}")
                self.comparison_text.append(f"Total anomalies: {len(anomalies)}")
                
                # Count by type
                type_counts = defaultdict(int)
                for anomaly in anomalies:
                    defect_type = anomaly.get('defect_type', 'Unknown')
                    type_counts[defect_type] += 1
                
                self.comparison_text.append("\nAnomalies by type:")
                for defect_type, count in type_counts.items():
                    self.comparison_text.append(f"- {defect_type}: {count}")
                
                # Count by severity
                severity_counts = defaultdict(int)
                for anomaly in anomalies:
                    severity = anomaly.get('severity', 'low')
                    severity_counts[severity] += 1
                
                self.comparison_text.append("\nAnomalies by severity:")
                for severity, count in severity_counts.items():
                    self.comparison_text.append(f"- {severity.capitalize()}: {count}")
    
    def on_add_inspection(self):
        """Add current data as new inspection."""
        if not self.anomalies:
            QMessageBox.warning(self, "Warning", "No anomalies to add")
            return
        
        # Create dialog for inspection metadata
        date_input, ok = QInputDialog.getText(
            self, "Inspection Date", 
            "Enter inspection date (YYYY-MM-DD):",
            text=date.today().strftime("%Y-%m-%d")
        )
        
        if not ok:
            return
        
        try:
            # Parse date
            inspection_date = datetime.strptime(date_input, "%Y-%m-%d").date()
            
            # Add to tracker
            self.anomaly_tracker.add_inspection(
                inspection_date, 
                self.anomalies,
                {
                    'filename': self.current_thermal_path
                }
            )
            
            # Update UI
            self._update_ui_state()
            
            QMessageBox.information(self, "Inspection Added", 
                                  f"Inspection for {date_input} added successfully")
            
        except ValueError:
            QMessageBox.critical(self, "Date Error", "Invalid date format")
        except Exception as e:
            QMessageBox.critical(self, "Add Error", f"Failed to add inspection: {str(e)}")
    
    def on_compare_inspections(self):
        """Compare selected inspection with previous one."""
        row = self.inspections_list.currentRow()
        if row < 0:
            return
        
        dates = self.anomaly_tracker.get_dates()
        if row < len(dates) and len(dates) >= 2:
            selected_date = dates[len(dates) - 1 - row]  # Reverse order
            
            # Find previous date
            prev_idx = dates.index(selected_date) - 1
            if prev_idx >= 0:
                prev_date = dates[prev_idx]
                
                # Compare inspections
                comparison = self.anomaly_tracker.compare_inspections(prev_date, selected_date)
                
                # Display comparison results
                self.comparison_text.clear()
                
                self.comparison_text.append(f"Comparison: {prev_date} vs {selected_date}")
                self.comparison_text.append(
                    f"Anomaly count change: {comparison['anomaly_diff']:+d} "
                    f"({comparison['stats1']['anomaly_count']} → {comparison['stats2']['anomaly_count']})"
                )
                self.comparison_text.append(
                    f"Power loss change: {comparison['power_loss_diff']:+.2f} kW "
                    f"({comparison['stats1']['power_loss_kw']:.2f} kW → {comparison['stats2']['power_loss_kw']:.2f} kW)"
                )
                
                # New issues
                self.comparison_text.append(f"\nNew issues: {len(comparison['new_issues'])}")
                for i, anomaly in enumerate(comparison['new_issues'][:5], 1):  # Show only first 5
                    defect_type = anomaly.get('defect_type', 'Unknown')
                    severity = anomaly.get('severity', 'low').capitalize()
                    component_id = anomaly.get('component_id', 'Unknown')
                    
                    self.comparison_text.append(
                        f"{i}. {defect_type} ({severity}) at {component_id}"
                    )
                
                if len(comparison['new_issues']) > 5:
                    self.comparison_text.append(f"... and {len(comparison['new_issues']) - 5} more")
                
                # Resolved issues
                self.comparison_text.append(f"\nResolved issues: {len(comparison['resolved_issues'])}")
                for i, anomaly in enumerate(comparison['resolved_issues'][:5], 1):  # Show only first 5
                    defect_type = anomaly.get('defect_type', 'Unknown')
                    severity = anomaly.get('severity', 'low').capitalize()
                    component_id = anomaly.get('component_id', 'Unknown')
                    
                    self.comparison_text.append(
                        f"{i}. {defect_type} ({severity}) at {component_id}"
                    )
                
                if len(comparison['resolved_issues']) > 5:
                    self.comparison_text.append(f"... and {len(comparison['resolved_issues']) - 5} more")
            else:
                self.comparison_text.clear()
                self.comparison_text.append("No previous inspection available for comparison")
    
    def on_browse_output(self):
        """Browse for output directory."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        
        if folder_path:
            self.output_dir_edit.setText(folder_path)
            
            # Update config
            self.config['reporting']['output_dir'] = folder_path
    
    def on_about(self):
        """Show about dialog."""
        about_text = """
        <h1>SolarEye - Thermal Analyzer</h1>
        <p>Version 1.0</p>
        <p>Advanced thermal analysis tool for solar PV installations</p>
        <p>Copyright (c) 2025</p>
        <p>Features:</p>
        <ul>
            <li>Thermal anomaly detection</li>
            <li>ML-based defect classification</li>
            <li>CAD overlay registration</li>
            <li>Site component identification</li>
            <li>Historical analysis</li>
            <li>Reporting and visualization</li>
        </ul>
        """
        
        QMessageBox.about(self, "About SolarEye", about_text)


def main():
    """Main application entry point with error logging."""
    try:
        logger.info("Starting application")
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle("Fusion")
        
        # Create and show main window
        main_window = MainWindow()
        main_window.show()
        
        logger.info("Application initialized successfully")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.critical(f"Fatal application error: {str(e)}\n{error_details}")
        
        # Try to show error dialog if possible
        try:
            QMessageBox.critical(
                None, 
                "Fatal Error", 
                f"A fatal error occurred:\n\n{str(e)}\n\nA detailed log has been saved to:\n{log_filename}"
            )
        except:
            # If dialog fails, print to console
            print(f"FATAL ERROR: {str(e)}")
            print(f"See log file for details: {log_filename}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()