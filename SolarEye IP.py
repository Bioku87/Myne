import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import datetime
import time

# Constants based on SolarEye Analyzer
BG_COLOR = "#041705"
BTN_BG = "#0A340A" 
BTN_FG = "#7CFC00"
FONT = ("Arial", 10, "bold")

# Defect types from SolarEye Analyzer
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
    'Sun reflection',  # Added sun reflection as a defect type
    'Unknown anomaly'
]

# Component types
COMPONENT_TYPES = [
    'Module',
    'String',
    'Combiner',
    'Tracker',
    'Substring'
]

# Severity levels
SEVERITY_LEVELS = ['low', 'medium', 'high']

# Pattern types
PATTERN_TYPES = ['Single', 'Identical', 'Complex']

# Thermal color palettes (OpenCV colormaps)
COLOR_PALETTES = {
    'FLIR (Inferno)': cv2.COLORMAP_INFERNO,
    'Rainbow': cv2.COLORMAP_RAINBOW,
    'Jet': cv2.COLORMAP_JET,
    'Hot': cv2.COLORMAP_HOT,
    'Plasma': cv2.COLORMAP_PLASMA,
    'Viridis': cv2.COLORMAP_VIRIDIS,
    'Parula': cv2.COLORMAP_PARULA
}

class Redirector:
    def __init__(self, widget):
        self.widget = widget
    def write(self, msg):
        if self.widget:
            self.widget.insert(tk.END, msg)
            self.widget.see(tk.END)
        try: 
            sys.__stdout__.write(msg)
        except: 
            pass
    def flush(self): 
        pass

class AnomalyMarker:
    def __init__(self, canvas, x, y, radius=10):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.radius = radius
        self.id = canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            outline="yellow", width=2, tags="anomaly"
        )
        self.box_id = None
        self.bbox = None
        self.text_id = None
        self.data = {
            'id': id(self),
            'centroid': (x, y),
            'area': np.pi * (radius ** 2),  # Approximate area in pixels
            'delta_t': 0.0,  # Will be set based on image data
            'max_temp': 0.0,
            'mean_temp': 0.0,
            'aspect_ratio': 1.0,  # Will be updated based on bbox
            'component_type': 'Module',
            'defect_type': 'Potential hot spot',
            'severity': 'medium',
            'pattern': 'Single',
            'perimeter': 2 * np.pi * radius,  # Approximate perimeter
            'circularity': 1.0,  # Perfect circle initially
            'solidity': 0.9,  # Solid by default
            'power_loss_kw': 0.0,
            'is_sun_reflection': False  # Flag for sun reflections
        }
        
    def set_bbox(self, x1, y1, x2, y2):
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Delete old box if exists
        if self.box_id:
            self.canvas.delete(self.box_id)
            
        # Create new box
        self.box_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="yellow", width=1, dash=(5, 5), tags="anomaly_box"
        )
        
        # Update bbox data
        self.bbox = (x1, y1, x2-x1, y2-y1)  # x, y, width, height
        
        # Update related properties
        width, height = x2-x1, y2-y1
        self.data['area'] = width * height
        self.data['aspect_ratio'] = width / max(height, 1)  # Avoid division by zero
        self.data['perimeter'] = 2 * (width + height)

    def update_text(self):
        if self.text_id:
            self.canvas.delete(self.text_id)
            
        self.text_id = self.canvas.create_text(
            self.x, self.y - self.radius - 10,
            text=f"{self.data['defect_type']} ({self.data['severity']})",
            fill="white", font=FONT, tags="anomaly_label"
        )
        
    def calculate_thermal_properties(self, thermal_data):
        """Calculate thermal properties based on the image data"""
        if thermal_data is None or self.bbox is None:
            return
            
        x, y, w, h = self.bbox
        x, y = int(x), int(y)
        w, h = int(w), int(h)
        
        # Ensure within bounds
        height, width = thermal_data.shape[:2]
        x = max(0, min(x, width-1))
        y = max(0, min(y, height-1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w <= 0 or h <= 0:
            return
            
        # Extract region
        region = thermal_data[y:y+h, x:x+w]
        
        if region.size == 0:
            return
            
        # Calculate statistics
        self.data['max_temp'] = float(np.max(region))
        self.data['mean_temp'] = float(np.mean(region))
        
        # Calculate delta T (difference from surrounding area)
        # Get a slightly larger region for background
        bg_x = max(0, x - 10)
        bg_y = max(0, y - 10)
        bg_w = min(w + 20, width - bg_x)
        bg_h = min(h + 20, height - bg_y)
        
        if bg_w <= 0 or bg_h <= 0:
            return
            
        # Create a mask for the background (everything except the anomaly)
        bg_region = thermal_data[bg_y:bg_y+bg_h, bg_x:bg_x+bg_w]
        mask = np.ones_like(bg_region, dtype=bool)
        
        # Adjust anomaly coordinates relative to background region
        rel_x, rel_y = x - bg_x, y - bg_y
        mask_w, mask_h = min(w, bg_w - rel_x), min(h, bg_h - rel_y)
        
        if mask_w <= 0 or mask_h <= 0 or rel_x >= bg_w or rel_y >= bg_h:
            return
            
        mask[rel_y:rel_y+mask_h, rel_x:rel_x+mask_w] = False
        
        # Calculate background temperature
        if np.any(mask):
            bg_temp = np.mean(bg_region[mask])
            self.data['delta_t'] = float(self.data['mean_temp'] - bg_temp)
        else:
            self.data['delta_t'] = 0.0
            
        # Calculate power loss based on severity and component type (skip for sun reflections)
        if not self.data['is_sun_reflection']:
            self.calculate_power_loss()
        
    def calculate_power_loss(self):
        """Calculate estimated power loss based on component type and severity"""
        # Default module power (W)
        module_power = 100
        
        # Affected modules
        if self.data['component_type'] == 'Module':
            affected_modules = 1
        elif self.data['component_type'] == 'String':
            affected_modules = 16  # Typical string has 16 modules
        elif self.data['component_type'] == 'Combiner':
            affected_modules = 160  # Typical combiner has 10 strings
        elif self.data['component_type'] == 'Tracker':
            affected_modules = 80  # Typical tracker might have 80 modules
        else:  # Substring
            affected_modules = 0.25  # Quarter of a module
            
        # Severity factor
        if self.data['severity'] == 'high':
            severity_factor = 0.9
        elif self.data['severity'] == 'medium':
            severity_factor = 0.5
        else:  # low
            severity_factor = 0.1
            
        # Calculate power loss in kW
        self.data['power_loss_kw'] = (affected_modules * module_power * severity_factor) / 1000
        
    def set_color_by_severity(self):
        """Update the marker color based on severity or type"""
        if self.data['is_sun_reflection'] or self.data['defect_type'] == 'Sun reflection':
            # Sun reflections are blue
            color = "cyan"
        elif self.data['severity'] == 'high':
            color = "red"
        elif self.data['severity'] == 'medium':
            color = "orange"
        else:  # low
            color = "yellow"
            
        self.canvas.itemconfig(self.id, outline=color)
        if self.box_id:
            self.canvas.itemconfig(self.box_id, outline=color)
            
    def set_as_sun_reflection(self):
        """Mark this anomaly as a sun reflection"""
        self.data['is_sun_reflection'] = True
        self.data['defect_type'] = 'Sun reflection'
        self.data['power_loss_kw'] = 0.0  # No power loss for reflections
        self.set_color_by_severity()
        self.update_text()

class TrainingDataGenerator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SolarEye Training Data Generator")
        self.configure(bg=BG_COLOR)
        
        # Set icon if available
        try:
            blank = tk.PhotoImage()
            self.iconphoto(False, blank)
        except:
            pass
            
        # Initialize variables
        self.current_image_path = None
        self.thermal_data = None
        self.display_image = None
        self.anomalies = []
        self.current_anomaly = None
        self.drawing_bbox = False
        self.start_x, self.start_y = 0, 0
        self.current_temp_range = (0, 100)  # Default temperature range
        self.use_fahrenheit = True  # Default to Fahrenheit
        self.current_palette = cv2.COLORMAP_INFERNO  # Default palette
        
        # Image scaling factors
        self.image_scale = 1.0
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        
        # Folder navigation variables
        self.input_folder = None
        self.image_files = []
        self.current_image_index = -1
        
        # Auto looping variables
        self.auto_loop_active = False
        self.loop_interval = 1.0  # Default interval in seconds
        self.loop_timer_id = None
        
        # Image data storage for marked images
        self.image_annotations = {}  # {image_path: [anomaly_data, ...]}
        
        # History for undo functionality
        self.history = []  # List of (action, data) tuples
        
        # Set up UI
        self.create_ui()
        
        # Bind keyboard events
        self.bind("<Left>", self.prev_image)
        self.bind("<Right>", self.next_image)
        self.bind("<Down>", self.toggle_auto_loop)
        self.bind("<Up>", self.save_current_image)
        self.bind("<Control-z>", self.undo_last_action)
        
        # Redirect stdout to log
        sys.stdout = Redirector(self.log)
        
        # Print welcome message
        print("SolarEye Training Data Generator")
        print("Created to generate training data for SolarEye Analyzer")
        print("1. Load input folder with thermal images")
        print("2. Mark anomalies on each image")
        print("3. Use keyboard controls: ← → to navigate, ↓ to toggle auto loop, ↑ to save, Ctrl+Z to undo")
        print("4. Export the data in CSV/Excel format when complete")
        
    def create_ui(self):
        # Create overall layout
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Create left panel (controls)
        left_frame = tk.Frame(self, bg=BG_COLOR, width=300)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.pack_propagate(False)
        self.create_control_panel(left_frame)
        
        # Create right panel (image and canvas)
        right_frame = tk.Frame(self, bg=BG_COLOR)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.create_image_panel(right_frame)
        
        # Create bottom panel (log)
        bottom_frame = tk.Frame(self, bg=BG_COLOR, height=100)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.create_log_panel(bottom_frame)
        
        # Set minimum window size
        self.minsize(900, 700)
        
    def create_control_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        
        # File controls
        file_frame = tk.LabelFrame(parent, text="File Operations", bg=BG_COLOR, fg=BTN_FG, font=FONT)
        file_frame.pack(fill="x", padx=5, pady=5)
        
        load_folder_btn = ttk.Button(file_frame, text="Load Input Folder", command=self.load_input_folder)
        load_folder_btn.pack(fill="x", padx=5, pady=5)
        
        load_btn = ttk.Button(file_frame, text="Load Single Image", command=self.load_image)
        load_btn.pack(fill="x", padx=5, pady=5)
        
        export_btn = ttk.Button(file_frame, text="Export Training Data", command=self.export_data)
        export_btn.pack(fill="x", padx=5, pady=5)
        
        # Navigation controls
        nav_frame = tk.LabelFrame(parent, text="Navigation", bg=BG_COLOR, fg=BTN_FG, font=FONT)
        nav_frame.pack(fill="x", padx=5, pady=5)
        
        nav_btn_frame = tk.Frame(nav_frame, bg=BG_COLOR)
        nav_btn_frame.pack(fill="x", padx=5, pady=5)
        
        prev_btn = ttk.Button(nav_btn_frame, text="← Prev", command=lambda: self.prev_image(None))
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = ttk.Button(nav_btn_frame, text="Next →", command=lambda: self.next_image(None))
        next_btn.pack(side=tk.RIGHT, padx=5)
        
        # Image counter
        self.image_counter_var = tk.StringVar(value="Image 0/0")
        counter_label = tk.Label(nav_frame, textvariable=self.image_counter_var, bg=BG_COLOR, fg=BTN_FG, font=FONT)
        counter_label.pack(fill="x", padx=5, pady=5)
        
        # Auto loop controls
        loop_frame = tk.Frame(nav_frame, bg=BG_COLOR)
        loop_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(loop_frame, text="Loop Interval:", bg=BG_COLOR, fg=BTN_FG, font=FONT).pack(side=tk.LEFT)
        
        self.loop_interval_var = tk.DoubleVar(value=1.0)
        interval_spinner = ttk.Spinbox(
            loop_frame, 
            from_=0.1, 
            to=2.0, 
            increment=0.1, 
            textvariable=self.loop_interval_var,
            width=5
        )
        interval_spinner.pack(side=tk.LEFT, padx=5)
        
        tk.Label(loop_frame, text="sec", bg=BG_COLOR, fg=BTN_FG, font=FONT).pack(side=tk.LEFT)
        
        self.loop_btn = ttk.Button(nav_frame, text="Start Auto Loop (↓)", command=self.toggle_auto_loop)
        self.loop_btn.pack(fill="x", padx=5, pady=5)
        
        self.save_btn = ttk.Button(nav_frame, text="Save Current Image (↑)", command=self.save_current_image)
        self.save_btn.pack(fill="x", padx=5, pady=5)
        
        undo_btn = ttk.Button(nav_frame, text="Undo Last Action (Ctrl+Z)", command=lambda: self.undo_last_action(None))
        undo_btn.pack(fill="x", padx=5, pady=5)
        
        # Thermal display settings
        thermal_frame = tk.LabelFrame(parent, text="Display Settings", bg=BG_COLOR, fg=BTN_FG, font=FONT)
        thermal_frame.pack(fill="x", padx=5, pady=5)
        
        # Color palette selection
        tk.Label(thermal_frame, text="Color Palette:", bg=BG_COLOR, fg=BTN_FG).pack(anchor="w", padx=5)
        self.palette_var = tk.StringVar(value="FLIR (Inferno)")
        palette_combo = ttk.Combobox(thermal_frame, textvariable=self.palette_var, values=list(COLOR_PALETTES.keys()))
        palette_combo.pack(fill="x", padx=5, pady=2)
        palette_combo.bind("<<ComboboxSelected>>", self.change_color_palette)
        
        # Temperature unit selection
        unit_frame = tk.Frame(thermal_frame, bg=BG_COLOR)
        unit_frame.pack(fill="x", padx=5, pady=5)
        
        self.temp_unit_var = tk.StringVar(value="°F")
        fahrenheit_radio = ttk.Radiobutton(unit_frame, text="Fahrenheit (°F)", variable=self.temp_unit_var, value="°F", command=self.change_temp_unit)
        fahrenheit_radio.pack(anchor="w")
        
        celsius_radio = ttk.Radiobutton(unit_frame, text="Celsius (°C)", variable=self.temp_unit_var, value="°C", command=self.change_temp_unit)
        celsius_radio.pack(anchor="w")
        
        # Thermal calibration
        cal_frame = tk.LabelFrame(parent, text="Temperature Calibration", bg=BG_COLOR, fg=BTN_FG, font=FONT)
        cal_frame.pack(fill="x", padx=5, pady=5)
        
        min_temp_frame = tk.Frame(cal_frame, bg=BG_COLOR)
        min_temp_frame.pack(fill="x", padx=5, pady=2)
        tk.Label(min_temp_frame, text="Min Temp:", bg=BG_COLOR, fg=BTN_FG).pack(side=tk.LEFT)
        self.min_temp_var = tk.StringVar(value="32")  # Default in °F
        min_temp_entry = ttk.Entry(min_temp_frame, textvariable=self.min_temp_var, width=8)
        min_temp_entry.pack(side=tk.LEFT, padx=5)
        self.min_temp_unit_label = tk.Label(min_temp_frame, text="°F", bg=BG_COLOR, fg=BTN_FG)
        self.min_temp_unit_label.pack(side=tk.LEFT)
        
        max_temp_frame = tk.Frame(cal_frame, bg=BG_COLOR)
        max_temp_frame.pack(fill="x", padx=5, pady=2)
        tk.Label(max_temp_frame, text="Max Temp:", bg=BG_COLOR, fg=BTN_FG).pack(side=tk.LEFT)
        self.max_temp_var = tk.StringVar(value="212")  # Default in °F
        max_temp_entry = ttk.Entry(max_temp_frame, textvariable=self.max_temp_var, width=8)
        max_temp_entry.pack(side=tk.LEFT, padx=5)
        self.max_temp_unit_label = tk.Label(max_temp_frame, text="°F", bg=BG_COLOR, fg=BTN_FG)
        self.max_temp_unit_label.pack(side=tk.LEFT)
        
        apply_cal_btn = ttk.Button(cal_frame, text="Apply Calibration", command=self.apply_calibration)
        apply_cal_btn.pack(fill="x", padx=5, pady=5)
        
        # Auto-detect full temperature range button
        detect_range_btn = ttk.Button(cal_frame, text="Auto-Detect Temperature Range", command=self.auto_detect_temp_range)
        detect_range_btn.pack(fill="x", padx=5, pady=5)
        
        # Ambient temperature
        ambient_frame = tk.Frame(cal_frame, bg=BG_COLOR)
        ambient_frame.pack(fill="x", padx=5, pady=2)
        tk.Label(ambient_frame, text="Ambient Temp:", bg=BG_COLOR, fg=BTN_FG).pack(side=tk.LEFT)
        self.ambient_temp_var = tk.StringVar(value="77")  # Default in °F
        ambient_temp_entry = ttk.Entry(ambient_frame, textvariable=self.ambient_temp_var, width=8)
        ambient_temp_entry.pack(side=tk.LEFT, padx=5)
        self.ambient_temp_unit_label = tk.Label(ambient_frame, text="°F", bg=BG_COLOR, fg=BTN_FG)
        self.ambient_temp_unit_label.pack(side=tk.LEFT)
        
        # Auto detection
        detection_frame = tk.LabelFrame(parent, text="Auto Detection", bg=BG_COLOR, fg=BTN_FG, font=FONT)
        detection_frame.pack(fill="x", padx=5, pady=5)
        
        threshold_frame = tk.Frame(detection_frame, bg=BG_COLOR)
        threshold_frame.pack(fill="x", padx=5, pady=2)
        tk.Label(threshold_frame, text="Temperature Threshold:", bg=BG_COLOR, fg=BTN_FG).pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="20")  # Delta T in degrees
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=8)
        threshold_entry.pack(side=tk.LEFT, padx=5)
        self.threshold_unit_label = tk.Label(threshold_frame, text="°F", bg=BG_COLOR, fg=BTN_FG)
        self.threshold_unit_label.pack(side=tk.LEFT)
        
        detect_btn = ttk.Button(detection_frame, text="Auto Detect Hotspots", command=self.auto_detect_hotspots)
        detect_btn.pack(fill="x", padx=5, pady=5)
        
        # Sun reflection option
        refl_frame = tk.Frame(detection_frame, bg=BG_COLOR)
        refl_frame.pack(fill="x", padx=5, pady=5)
        
        mark_sun_refl_btn = ttk.Button(refl_frame, text="Mark as Sun Reflection", command=self.mark_as_sun_reflection)
        mark_sun_refl_btn.pack(fill="x", padx=5, pady=5)
        
        # Anomaly properties
        props_frame = tk.LabelFrame(parent, text="Anomaly Properties", bg=BG_COLOR, fg=BTN_FG, font=FONT)
        props_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Defect type
        tk.Label(props_frame, text="Defect Type:", bg=BG_COLOR, fg=BTN_FG).pack(anchor="w", padx=5)
        self.defect_type_var = tk.StringVar()
        defect_combo = ttk.Combobox(props_frame, textvariable=self.defect_type_var, values=DEFECT_TYPES)
        defect_combo.pack(fill="x", padx=5, pady=2)
        defect_combo.current(5)  # Default to "Potential hot spot"
        
        # Component type
        tk.Label(props_frame, text="Component Type:", bg=BG_COLOR, fg=BTN_FG).pack(anchor="w", padx=5)
        self.component_type_var = tk.StringVar()
        component_combo = ttk.Combobox(props_frame, textvariable=self.component_type_var, values=COMPONENT_TYPES)
        component_combo.pack(fill="x", padx=5, pady=2)
        component_combo.current(0)  # Default to "Module"
        
        # Severity
        tk.Label(props_frame, text="Severity:", bg=BG_COLOR, fg=BTN_FG).pack(anchor="w", padx=5)
        self.severity_var = tk.StringVar()
        severity_combo = ttk.Combobox(props_frame, textvariable=self.severity_var, values=SEVERITY_LEVELS)
        severity_combo.pack(fill="x", padx=5, pady=2)
        severity_combo.current(1)  # Default to "medium"
        
        # Pattern
        tk.Label(props_frame, text="Pattern:", bg=BG_COLOR, fg=BTN_FG).pack(anchor="w", padx=5)
        self.pattern_var = tk.StringVar()
        pattern_combo = ttk.Combobox(props_frame, textvariable=self.pattern_var, values=PATTERN_TYPES)
        pattern_combo.pack(fill="x", padx=5, pady=2)
        pattern_combo.current(0)  # Default to "Single"
        
        # Apply button
        apply_btn = ttk.Button(props_frame, text="Apply to Selected", command=self.apply_properties)
        apply_btn.pack(fill="x", padx=5, pady=5)
        
        # Deletion button
        delete_btn = ttk.Button(props_frame, text="Delete Selected", command=self.delete_selected)
        delete_btn.pack(fill="x", padx=5, pady=5)
        
        # Feature preview
        preview_frame = tk.LabelFrame(parent, text="Feature Preview", bg=BG_COLOR, fg=BTN_FG, font=FONT)
        preview_frame.pack(fill="x", padx=5, pady=5)
        
        self.preview_text = tk.Text(preview_frame, height=8, width=30, bg="black", fg="green", font=("Courier", 9))
        self.preview_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_image_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Canvas for image display and interaction
        self.canvas_frame = tk.Frame(parent, bg="black")
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Placeholder text
        self.canvas.create_text(
            self.canvas.winfo_reqwidth() // 2, 
            self.canvas.winfo_reqheight() // 2,
            text="Load an input folder or image to begin",
            fill="white",
            font=FONT,
            tags="placeholder"
        )
        
        # Colorbar frame with more height for temperature scale
        self.colorbar_frame = tk.Frame(parent, height=70, bg=BG_COLOR)
        self.colorbar_frame.grid(row=1, column=0, sticky="ew")
        self.colorbar = None
        
    def create_log_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        self.log = tk.Text(parent, height=6, bg="black", fg="green", font=("Courier", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(parent, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.config(yscrollcommand=scrollbar.set)
    
    def auto_detect_temp_range(self):
        """Auto-detect the temperature range from the image data"""
        if self.thermal_data is None:
            messagebox.showinfo("Info", "No image loaded")
            return
            
        try:
            # Get min and max pixel values from the image
            min_val = np.min(self.thermal_data)
            max_val = np.max(self.thermal_data)
            
            # Convert to temperature values based on current calibration
            old_min, old_max = self.current_temp_range
            temp_range = old_max - old_min
            
            # Normalize to temperature range
            min_temp = old_min + (min_val - np.min(self.thermal_data)) / (np.max(self.thermal_data) - np.min(self.thermal_data)) * temp_range
            max_temp = old_min + (max_val - np.min(self.thermal_data)) / (np.max(self.thermal_data) - np.min(self.thermal_data)) * temp_range
            
            # Add a little padding
            min_temp = min_temp - 5
            max_temp = max_temp + 5
            
            # Convert to Fahrenheit if needed
            if self.use_fahrenheit:
                min_temp_display = self.celsius_to_fahrenheit(min_temp)
                max_temp_display = self.celsius_to_fahrenheit(max_temp)
            else:
                min_temp_display = min_temp
                max_temp_display = max_temp
                
            # Update UI
            self.min_temp_var.set(f"{min_temp_display:.1f}")
            self.max_temp_var.set(f"{max_temp_display:.1f}")
            
            # Apply the new calibration
            self.apply_calibration()
            
            unit = "°F" if self.use_fahrenheit else "°C"
            print(f"Auto-detected temperature range: {min_temp_display:.1f}{unit} to {max_temp_display:.1f}{unit}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-detect temperature range: {str(e)}")
            print(f"Error auto-detecting temperature range: {str(e)}")
    
    def mark_as_sun_reflection(self):
        """Mark the selected anomaly as a sun reflection"""
        if not self.current_anomaly:
            messagebox.showinfo("Info", "No anomaly selected")
            return
            
        # Save old properties for undo
        old_properties = self.current_anomaly.data.copy()
        self.history.append(("modify_properties", (self.current_anomaly, old_properties)))
        
        # Set as sun reflection
        self.current_anomaly.set_as_sun_reflection()
        
        # Update preview
        self.update_preview()
        
        print(f"Marked anomaly as Sun Reflection")
    
    def celsius_to_fahrenheit(self, temp_c):
        """Convert Celsius to Fahrenheit"""
        return (temp_c * 9/5) + 32
    
    def fahrenheit_to_celsius(self, temp_f):
        """Convert Fahrenheit to Celsius"""
        return (temp_f - 32) * 5/9
    
    def change_temp_unit(self):
        """Handle temperature unit change"""
        new_unit = self.temp_unit_var.get()
        
        # Update labels
        self.min_temp_unit_label.config(text=new_unit)
        self.max_temp_unit_label.config(text=new_unit)
        self.ambient_temp_unit_label.config(text=new_unit)
        self.threshold_unit_label.config(text=new_unit)
        
        # Convert values if needed
        if new_unit == "°F" and not self.use_fahrenheit:
            # Convert from C to F
            self.use_fahrenheit = True
            try:
                min_temp = float(self.min_temp_var.get())
                max_temp = float(self.max_temp_var.get())
                ambient_temp = float(self.ambient_temp_var.get())
                threshold = float(self.threshold_var.get())
                
                self.min_temp_var.set(f"{self.celsius_to_fahrenheit(min_temp):.1f}")
                self.max_temp_var.set(f"{self.celsius_to_fahrenheit(max_temp):.1f}")
                self.ambient_temp_var.set(f"{self.celsius_to_fahrenheit(ambient_temp):.1f}")
                self.threshold_var.set(f"{threshold * 9/5:.1f}")  # Convert delta T
            except ValueError:
                pass
                
        elif new_unit == "°C" and self.use_fahrenheit:
            # Convert from F to C
            self.use_fahrenheit = False
            try:
                min_temp = float(self.min_temp_var.get())
                max_temp = float(self.max_temp_var.get())
                ambient_temp = float(self.ambient_temp_var.get())
                threshold = float(self.threshold_var.get())
                
                self.min_temp_var.set(f"{self.fahrenheit_to_celsius(min_temp):.1f}")
                self.max_temp_var.set(f"{self.fahrenheit_to_celsius(max_temp):.1f}")
                self.ambient_temp_var.set(f"{self.fahrenheit_to_celsius(ambient_temp):.1f}")
                self.threshold_var.set(f"{threshold * 5/9:.1f}")  # Convert delta T
            except ValueError:
                pass
        
        # Update display
        self.apply_calibration()
        
        # Update preview
        if self.current_anomaly:
            self.update_preview()
        
    def change_color_palette(self, event=None):
        """Change the thermal color palette"""
        palette_name = self.palette_var.get()
        self.current_palette = COLOR_PALETTES[palette_name]
        
        # Update image display
        self.update_image_display()
        
        # Update colorbar
        self.create_colorbar()
        
        print(f"Changed color palette to {palette_name}")
    
    def load_input_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Input Folder with Thermal Images"
        )
        
        if not folder_path:
            return
            
        try:
            # Find all image files in the folder
            self.input_folder = folder_path
            self.image_files = self.get_image_files(folder_path)
            
            if not self.image_files:
                messagebox.showinfo("Info", "No image files found in the selected folder")
                return
                
            # Reset current image index
            self.current_image_index = -1
            
            # Update image counter
            self.update_image_counter()
            
            # Load first image
            self.next_image(None)
            
            print(f"Loaded input folder: {folder_path}")
            print(f"Found {len(self.image_files)} image files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load input folder: {str(e)}")
            print(f"Error loading input folder: {str(e)}")
    
    def get_image_files(self, folder_path):
        """Get all image files in the folder"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_files = []
        
        for file in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(folder_path, file))
                
        return image_files
    
    def update_image_counter(self):
        """Update the image counter text"""
        if self.image_files:
            current = self.current_image_index + 1 if self.current_image_index >= 0 else 0
            total = len(self.image_files)
            self.image_counter_var.set(f"Image {current}/{total}")
        else:
            self.image_counter_var.set("Image 0/0")
    
    def next_image(self, event=None):
        """Load the next image in the folder"""
        if not self.image_files:
            return
            
        # Save annotations for current image
        self.save_current_annotations()
        
        # Increment index
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        
        # Load the image
        image_path = self.image_files[self.current_image_index]
        self.load_image_file(image_path)
        
        # Update counter
        self.update_image_counter()
        
        # Restore any saved annotations
        self.restore_annotations(image_path)
        
        # Clear history for the new image
        self.history = []
        
        return "break"  # Prevent default event handling
    
    def prev_image(self, event=None):
        """Load the previous image in the folder"""
        if not self.image_files:
            return
            
        # Save annotations for current image
        self.save_current_annotations()
        
        # Decrement index
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        
        # Load the image
        image_path = self.image_files[self.current_image_index]
        self.load_image_file(image_path)
        
        # Update counter
        self.update_image_counter()
        
        # Restore any saved annotations
        self.restore_annotations(image_path)
        
        # Clear history for the new image
        self.history = []
        
        return "break"  # Prevent default event handling
    
    def toggle_auto_loop(self, event=None):
        """Toggle auto looping through images"""
        if self.auto_loop_active:
            # Stop auto loop
            self.auto_loop_active = False
            if self.loop_timer_id:
                self.after_cancel(self.loop_timer_id)
                self.loop_timer_id = None
            self.loop_btn.config(text="Start Auto Loop (↓)")
            print("Auto loop stopped")
        else:
            # Start auto loop
            if not self.image_files:
                messagebox.showinfo("Info", "No images loaded. Please load an input folder first.")
                return
                
            self.auto_loop_active = True
            self.loop_btn.config(text="Stop Auto Loop (↓)")
            print(f"Auto loop started with interval {self.loop_interval_var.get()} seconds")
            self.auto_loop_next()
        
        return "break"  # Prevent default event handling
    
    def auto_loop_next(self):
        """Process next image in auto loop"""
        if not self.auto_loop_active:
            return
            
        # Load next image
        self.next_image()
        
        # Schedule next loop
        interval_ms = int(self.loop_interval_var.get() * 1000)
        self.loop_timer_id = self.after(interval_ms, self.auto_loop_next)
    
    def save_current_image(self, event=None):
        """Save current image annotations"""
        if not self.current_image_path:
            messagebox.showinfo("Info", "No image loaded")
            return
            
        self.save_current_annotations()
        messagebox.showinfo("Saved", f"Annotations saved for {os.path.basename(self.current_image_path)}")
        print(f"Saved annotations for {os.path.basename(self.current_image_path)}")
        
        return "break"  # Prevent default event handling
    
    def save_current_annotations(self):
        """Save annotations for the current image"""
        if not self.current_image_path or not self.anomalies:
            return
            
        # Extract data from anomalies
        annotations = []
        for anomaly in self.anomalies:
            annotation = anomaly.data.copy()
            
            # Convert centroid to a tuple
            x, y = annotation['centroid']
            annotation['centroid'] = (float(x), float(y))
            
            # Add bbox if exists
            if anomaly.bbox:
                annotation['bbox'] = anomaly.bbox
                
            annotations.append(annotation)
            
        # Save to dictionary
        self.image_annotations[self.current_image_path] = annotations
    
    def restore_annotations(self, image_path):
        """Restore saved annotations for an image"""
        if image_path not in self.image_annotations:
            return
            
        # Clear existing anomalies
        self.clear_anomalies()
        
        # Recreate anomalies from saved data
        annotations = self.image_annotations[image_path]
        for annotation in annotations:
            # Create marker
            x, y = annotation['centroid']
            anomaly = AnomalyMarker(self.canvas, x, y)
            
            # Set data
            for key, value in annotation.items():
                if key != 'centroid' and key != 'bbox':
                    anomaly.data[key] = value
            
            # Set bbox if exists
            if 'bbox' in annotation:
                x, y, w, h = annotation['bbox']
                anomaly.set_bbox(x, y, x+w, y+h)
            
            # Update text and color
            anomaly.update_text()
            anomaly.set_color_by_severity()
            
            # Add to list
            self.anomalies.append(anomaly)
        
        print(f"Restored {len(annotations)} annotations for {os.path.basename(image_path)}")
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Thermal Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("FLIR Files", "*.seq"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        # Save annotations for current image
        self.save_current_annotations()
        
        # Load the selected image
        self.load_image_file(file_path)
        
        # Reset folder navigation
        self.input_folder = None
        self.image_files = []
        self.current_image_index = -1
        self.update_image_counter()
        
        # Restore any saved annotations
        self.restore_annotations(file_path)
        
        # Clear history for the new image
        self.history = []
    
    def load_image_file(self, file_path):
        try:
            # Try to load image
            self.current_image_path = file_path
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.tif', '.tiff']:
                # Handle 16-bit TIFF files
                self.thermal_data = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
            else:
                # Regular images
                self.thermal_data = cv2.imread(file_path)
                
                # Convert to grayscale if color
                if len(self.thermal_data.shape) == 3:
                    self.thermal_data = cv2.cvtColor(self.thermal_data, cv2.COLOR_BGR2GRAY)
            
            # Auto-detect temperature range if first image
            if self.current_image_index == 0 or self.current_image_index == -1:
                self.auto_detect_temp_range()
            else:
                # Display the image
                self.update_image_display()
            
            # Clear anomalies if not restoring
            if file_path not in self.image_annotations:
                self.clear_anomalies()
            
            # Create colorbar
            self.create_colorbar()
            
            print(f"Loaded image: {os.path.basename(file_path)}")
            print(f"Image size: {self.thermal_data.shape[1]}x{self.thermal_data.shape[0]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            print(f"Error loading image: {str(e)}")
        
    def update_image_display(self):
        if self.thermal_data is None:
            return
            
        # Clear canvas
        self.canvas.delete("all")
        
        # Get min and max temperatures for normalization
        min_temp, max_temp = self.current_temp_range
        
        # Normalize and apply colormap
        normalized = cv2.normalize(
            self.thermal_data,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )
        
        # Apply current colormap
        colored = cv2.applyColorMap(normalized, self.current_palette)
        
        # Convert to PIL format for tkinter
        pil_img = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        
        # Resize to fit canvas if needed
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_img.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Save scale factors for coordinate mapping
            self.scale_factor_x = scale
            self.scale_factor_y = scale
            self.image_scale = scale
            
            if new_width != img_width or new_height != img_height:
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.display_image = ImageTk.PhotoImage(pil_img)
        
        # Add to canvas
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)
        
        # Redraw anomalies
        self.redraw_anomalies()
        
    def create_colorbar(self):
        # Clear existing colorbar
        for widget in self.colorbar_frame.winfo_children():
            widget.destroy()
            
        # Create matplotlib figure for colorbar
        fig = Figure(figsize=(8, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create colorbar
        min_temp, max_temp = self.current_temp_range
        
        # Convert to display units if needed
        if self.use_fahrenheit:
            display_min = self.celsius_to_fahrenheit(min_temp)
            display_max = self.celsius_to_fahrenheit(max_temp)
        else:
            display_min = min_temp
            display_max = max_temp
            
        # Create colormap
        norm = plt.Normalize(min_temp, max_temp)
        
        # Map OpenCV colormap to matplotlib colormap
        if self.current_palette == cv2.COLORMAP_INFERNO:
            cmap = plt.cm.inferno
        elif self.current_palette == cv2.COLORMAP_JET:
            cmap = plt.cm.jet
        elif self.current_palette == cv2.COLORMAP_VIRIDIS:
            cmap = plt.cm.viridis
        elif self.current_palette == cv2.COLORMAP_PLASMA:
            cmap = plt.cm.plasma
        elif self.current_palette == cv2.COLORMAP_HOT:
            cmap = plt.cm.hot
        elif self.current_palette == cv2.COLORMAP_RAINBOW:
            cmap = plt.cm.rainbow
        elif self.current_palette == cv2.COLORMAP_PARULA:
            cmap = plt.cm.plasma  # Closest to parula
        else:
            cmap = plt.cm.inferno  # Default
        
        # Create the colorbar with more ticks    
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                        cax=ax, orientation='horizontal')
        
        # Create more detailed tick marks
        unit = "°F" if self.use_fahrenheit else "°C"
        cb.set_label(f'Temperature ({unit})')
        
        # Calculate number of ticks based on temperature range
        temp_range = display_max - display_min
        if temp_range > 100:
            tick_interval = 20
        elif temp_range > 50:
            tick_interval = 10
        elif temp_range > 20:
            tick_interval = 5
        else:
            tick_interval = 2
            
        # Set custom ticks
        ticks = np.arange(
            np.floor(display_min / tick_interval) * tick_interval,
            np.ceil(display_max / tick_interval) * tick_interval + tick_interval,
            tick_interval
        )
        
        # Filter ticks to be within the range
        ticks = ticks[(ticks >= display_min) & (ticks <= display_max)]
        
        # Convert ticks back to Celsius for the norm if using Fahrenheit
        if self.use_fahrenheit:
            cb_ticks = [self.fahrenheit_to_celsius(t) for t in ticks]
            cb.set_ticks(cb_ticks)
            cb.set_ticklabels([f"{t:.1f}" for t in ticks])
        else:
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{t:.1f}" for t in ticks])
                
        # Set the colorbar size
        fig.subplots_adjust(bottom=0.3)
        
        # Add annotations for temperature shades
        levels = 5  # Number of temperature zones to label
        
        # Create a temperature table below the colorbar
        temp_step = (display_max - display_min) / levels
        
        temperature_ranges = [
            (display_min + i * temp_step, display_min + (i + 1) * temp_step) 
            for i in range(levels)
        ]
        
        # Add temperature range annotations below the colorbar
        for i, (t_min, t_max) in enumerate(temperature_ranges):
            # Skip annotations that are too crowded or outside the range
            if i % 2 == 0 or i == levels - 1:
                pos = i / levels
                ax.annotate(f"{t_min:.1f} - {t_max:.1f}{unit}", 
                            xy=(pos, -0.2),
                            xycoords='axes fraction',
                            ha='center', va='top',
                            fontsize=8)
        
        # Show actual image min/max
        if self.thermal_data is not None:
            try:
                # Calculate the actual min and max temperatures in the image
                img_min = np.min(self.thermal_data)
                img_max = np.max(self.thermal_data)
                
                # Convert to temperature values
                norm_min = (img_min - np.min(self.thermal_data)) / (np.max(self.thermal_data) - np.min(self.thermal_data))
                norm_max = (img_max - np.min(self.thermal_data)) / (np.max(self.thermal_data) - np.min(self.thermal_data))
                
                actual_min_temp = min_temp + norm_min * (max_temp - min_temp)
                actual_max_temp = min_temp + norm_max * (max_temp - min_temp)
                
                # Convert to display units
                if self.use_fahrenheit:
                    actual_min_temp = self.celsius_to_fahrenheit(actual_min_temp)
                    actual_max_temp = self.celsius_to_fahrenheit(actual_max_temp)
                
                # Add image min/max info
                fig.text(0.01, 0.01, f"Image Min: {actual_min_temp:.1f}{unit}", fontsize=8)
                fig.text(0.99, 0.01, f"Image Max: {actual_max_temp:.1f}{unit}", fontsize=8, ha='right')
                
            except Exception as e:
                print(f"Error calculating image min/max: {str(e)}")
                
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.colorbar_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def apply_calibration(self):
        try:
            min_temp = float(self.min_temp_var.get())
            max_temp = float(self.max_temp_var.get())
            
            if min_temp >= max_temp:
                messagebox.showerror("Error", "Min temperature must be less than max temperature")
                return
                
            # Convert to Celsius internally if using Fahrenheit
            if self.use_fahrenheit:
                min_temp = self.fahrenheit_to_celsius(min_temp)
                max_temp = self.fahrenheit_to_celsius(max_temp)
                
            self.current_temp_range = (min_temp, max_temp)
            
            # Update display
            self.update_image_display()
            
            # Update colorbar
            self.create_colorbar()
            
            unit = "°F" if self.use_fahrenheit else "°C"
            print(f"Applied temperature calibration: {self.min_temp_var.get()}{unit} to {self.max_temp_var.get()}{unit}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid temperature values")
    
    def undo_last_action(self, event=None):
        """Undo the last action"""
        if not self.history:
            messagebox.showinfo("Info", "Nothing to undo")
            return
            
        action, data = self.history.pop()
        
        if action == "add_anomaly":
            # Remove the last added anomaly
            if self.anomalies:
                anomaly = self.anomalies.pop()
                self.canvas.delete(anomaly.id)
                if anomaly.box_id:
                    self.canvas.delete(anomaly.box_id)
                if anomaly.text_id:
                    self.canvas.delete(anomaly.text_id)
                    
                # Clear current selection if it was the undone anomaly
                if self.current_anomaly == anomaly:
                    self.current_anomaly = None
                    self.preview_text.delete(1.0, tk.END)
                    
                print("Undid last anomaly addition")
                
        elif action == "delete_anomaly":
            # Restore the deleted anomaly
            anomaly_data = data
            
            # Create marker
            x, y = anomaly_data['centroid']
            anomaly = AnomalyMarker(self.canvas, x, y)
            
            # Set data
            for key, value in anomaly_data.items():
                if key != 'centroid' and key != 'bbox':
                    anomaly.data[key] = value
            
            # Set bbox if exists
            if 'bbox' in anomaly_data:
                x, y, w, h = anomaly_data['bbox']
                anomaly.set_bbox(x, y, x+w, y+h)
            
            # Update text and color
            anomaly.update_text()
            anomaly.set_color_by_severity()
            
            # Add to list
            self.anomalies.append(anomaly)
            
            print("Undid anomaly deletion")
            
        elif action == "modify_properties":
            # Restore previous properties
            anomaly, old_properties = data
            
            # Check if anomaly still exists
            if anomaly in self.anomalies:
                # Restore properties
                for key, value in old_properties.items():
                    anomaly.data[key] = value
                    
                # Update text
                anomaly.update_text()
                
                # Update color
                anomaly.set_color_by_severity()
                
                # Update preview if this is the current anomaly
                if self.current_anomaly == anomaly:
                    self.update_preview()
                    
                print("Undid property modification")
        
        return "break"  # Prevent default event handling
    
    def auto_detect_hotspots(self):
        """Automatically detect hotspots in the thermal image"""
        if self.thermal_data is None:
            messagebox.showinfo("Info", "No image loaded")
            return
            
        try:
            # Get threshold temperature difference
            threshold = float(self.threshold_var.get())
            
            # Convert to Celsius if using Fahrenheit
            if self.use_fahrenheit:
                threshold = threshold * 5/9  # Convert delta T from F to C
                
            # Get ambient temperature
            ambient = float(self.ambient_temp_var.get())
            if self.use_fahrenheit:
                ambient = self.fahrenheit_to_celsius(ambient)
                
            # Normalize image data to temperature scale
            min_temp, max_temp = self.current_temp_range
            temp_range = max_temp - min_temp
            
            # Create temperature map
            height, width = self.thermal_data.shape[:2]
            temp_map = np.zeros_like(self.thermal_data, dtype=np.float32)
            
            # Scale pixel values to temperature range
            for y in range(height):
                for x in range(width):
                    pixel_value = self.thermal_data[y, x]
                    normalized = (pixel_value - np.min(self.thermal_data)) / (np.max(self.thermal_data) - np.min(self.thermal_data))
                    temp_map[y, x] = min_temp + (normalized * temp_range)
            
            # Find areas with temperature above ambient + threshold
            hot_areas = np.where(temp_map > (ambient + threshold))
            
            if len(hot_areas[0]) == 0:
                messagebox.showinfo("Info", "No hotspots detected with current threshold")
                return
                
            # Use connected components to find distinct hotspots
            temp_binary = (temp_map > (ambient + threshold)).astype(np.uint8) * 255
            
            # Apply some morphology to clean up the binary image
            kernel = np.ones((3, 3), np.uint8)
            temp_binary = cv2.morphologyEx(temp_binary, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(temp_binary)
            
            # Skip the first component (background)
            detected_count = 0
            for i in range(1, num_labels):
                # Get component properties
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Filter out very small components (likely noise)
                if area < 10:
                    continue
                    
                # Get centroid
                cx, cy = centroids[i]
                
                # Get bounding box
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Apply scale factor for display
                cx = cx * self.scale_factor_x
                cy = cy * self.scale_factor_y
                x = x * self.scale_factor_x
                y = y * self.scale_factor_y
                w = w * self.scale_factor_x
                h = h * self.scale_factor_y
                
                # Create anomaly marker
                anomaly = self.create_anomaly_marker(int(cx), int(cy))
                
                # Set bounding box
                anomaly.set_bbox(x, y, x+w, y+h)
                
                # Calculate thermal properties
                anomaly.calculate_thermal_properties(self.thermal_data)
                
                # Set severity based on temperature difference
                if anomaly.data['delta_t'] < threshold/2:
                    anomaly.data['severity'] = 'low'
                elif anomaly.data['delta_t'] < threshold:
                    anomaly.data['severity'] = 'medium'
                else:
                    anomaly.data['severity'] = 'high'
                    
                # Update text and color
                anomaly.update_text()
                anomaly.set_color_by_severity()
                
                # Add to history for undo
                self.history.append(("add_anomaly", None))
                
                detected_count += 1
                
            if detected_count > 0:
                print(f"Auto-detected {detected_count} hotspots")
                messagebox.showinfo("Detection Complete", f"Detected {detected_count} hotspots")
            else:
                print("No significant hotspots detected")
                messagebox.showinfo("Info", "No significant hotspots detected with current threshold")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect hotspots: {str(e)}")
            print(f"Error detecting hotspots: {str(e)}")
    
    def on_canvas_click(self, event):
        if self.thermal_data is None:
            return
            
        # Start drawing bbox or add new marker
        self.start_x, self.start_y = event.x, event.y
        
        # Clear current selection
        self.current_anomaly = None
        
        # Check if clicked on an existing anomaly
        overlapping = self.canvas.find_overlapping(
            event.x - 5, event.y - 5, 
            event.x + 5, event.y + 5
        )
        
        for item_id in overlapping:
            tags = self.canvas.gettags(item_id)
            if "anomaly" in tags:
                # Find the corresponding anomaly
                for anomaly in self.anomalies:
                    if anomaly.id == item_id:
                        self.current_anomaly = anomaly
                        self.update_preview()
                        return
        
        # If no existing anomaly was clicked, start drawing a new one
        self.drawing_bbox = True
        self.temp_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, 
            self.start_x, self.start_y,
            outline="white", dash=(2, 2)
        )
        
    def on_canvas_drag(self, event):
        if self.drawing_bbox and self.temp_rect_id:
            # Update rectangle
            self.canvas.coords(self.temp_rect_id, self.start_x, self.start_y, event.x, event.y)
            
    def on_canvas_release(self, event):
        if self.drawing_bbox and self.temp_rect_id:
            # Finish drawing rectangle
            self.drawing_bbox = False
            
            # Get coordinates
            x1, y1 = self.start_x, self.start_y
            x2, y2 = event.x, event.y
            
            # Ensure minimum size
            if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                # Too small, create a marker instead
                self.canvas.delete(self.temp_rect_id)
                self.create_anomaly_marker(x1, y1)
            else:
                # Create a marker at the center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                anomaly = self.create_anomaly_marker(center_x, center_y)
                
                # Set bbox
                anomaly.set_bbox(x1, y1, x2, y2)
                
                # Delete temporary rectangle
                self.canvas.delete(self.temp_rect_id)
                
                # Calculate thermal properties
                if self.thermal_data is not None:
                    anomaly.calculate_thermal_properties(self.thermal_data)
                    
                # Update text
                anomaly.update_text()
                
            # Clean up
            self.temp_rect_id = None
            
    def create_anomaly_marker(self, x, y):
        # Create new anomaly marker
        anomaly = AnomalyMarker(self.canvas, x, y)
        self.anomalies.append(anomaly)
        
        # Add to history for undo
        self.history.append(("add_anomaly", None))
        
        # Set as current
        self.current_anomaly = anomaly
        
        # Set default properties
        anomaly.data['defect_type'] = self.defect_type_var.get()
        anomaly.data['component_type'] = self.component_type_var.get()
        anomaly.data['severity'] = self.severity_var.get()
        anomaly.data['pattern'] = self.pattern_var.get()
        
        # Update text
        anomaly.update_text()
        
        # Update color
        anomaly.set_color_by_severity()
        
        # Calculate thermal properties
        if self.thermal_data is not None:
            anomaly.calculate_thermal_properties(self.thermal_data)
            
        # Update preview
        self.update_preview()
        
        print(f"Created new anomaly at ({x}, {y})")
        
        return anomaly
        
    def clear_anomalies(self):
        # Remove all anomalies
        for anomaly in self.anomalies:
            self.canvas.delete(anomaly.id)
            if anomaly.box_id:
                self.canvas.delete(anomaly.box_id)
            if anomaly.text_id:
                self.canvas.delete(anomaly.text_id)
                
        self.anomalies = []
        self.current_anomaly = None
        self.preview_text.delete(1.0, tk.END)
        
        # Clear history
        self.history = []
        
    def redraw_anomalies(self):
        # Redraw all anomalies
        for anomaly in self.anomalies:
            # Redraw marker
            x, y = anomaly.x, anomaly.y
            radius = anomaly.radius
            
            anomaly.id = self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                outline="yellow" if not anomaly.data['is_sun_reflection'] else "cyan", 
                width=2, 
                tags="anomaly"
            )
            
            # Redraw box if exists
            if anomaly.bbox:
                x, y, w, h = anomaly.bbox
                anomaly.box_id = self.canvas.create_rectangle(
                    x, y, x + w, y + h,
                    outline="yellow" if not anomaly.data['is_sun_reflection'] else "cyan",
                    width=1, dash=(5, 5), tags="anomaly_box"
                )
                
            # Redraw text
            anomaly.update_text()
            
            # Update color
            anomaly.set_color_by_severity()
            
    def apply_properties(self):
        if not self.current_anomaly:
            messagebox.showinfo("Info", "No anomaly selected")
            return
            
        # Save old properties for undo
        old_properties = self.current_anomaly.data.copy()
        self.history.append(("modify_properties", (self.current_anomaly, old_properties)))
        
        # Apply properties from UI
        self.current_anomaly.data['defect_type'] = self.defect_type_var.get()
        self.current_anomaly.data['component_type'] = self.component_type_var.get()
        self.current_anomaly.data['severity'] = self.severity_var.get()
        self.current_anomaly.data['pattern'] = self.pattern_var.get()
        
        # Check if this is a sun reflection
        if self.current_anomaly.data['defect_type'] == 'Sun reflection':
            self.current_anomaly.data['is_sun_reflection'] = True
            self.current_anomaly.data['power_loss_kw'] = 0.0
        else:
            self.current_anomaly.data['is_sun_reflection'] = False
            # Recalculate power loss
            self.current_anomaly.calculate_power_loss()
        
        # Update text
        self.current_anomaly.update_text()
        
        # Update color
        self.current_anomaly.set_color_by_severity()
        
        # Update preview
        self.update_preview()
        
        print(f"Applied properties to anomaly {self.current_anomaly.data['id']}")
        
    def delete_selected(self):
        if not self.current_anomaly:
            messagebox.showinfo("Info", "No anomaly selected")
            return
            
        # Save anomaly data for undo
        anomaly_data = self.current_anomaly.data.copy()
        if self.current_anomaly.bbox:
            anomaly_data['bbox'] = self.current_anomaly.bbox
            
        self.history.append(("delete_anomaly", anomaly_data))
        
        # Remove from canvas
        self.canvas.delete(self.current_anomaly.id)
        if self.current_anomaly.box_id:
            self.canvas.delete(self.current_anomaly.box_id)
        if self.current_anomaly.text_id:
            self.canvas.delete(self.current_anomaly.text_id)
            
        # Remove from list
        self.anomalies.remove(self.current_anomaly)
        
        # Clear selection
        self.current_anomaly = None
        
        # Clear preview
        self.preview_text.delete(1.0, tk.END)
        
        print("Deleted selected anomaly")
        
    def update_preview(self):
        if not self.current_anomaly:
            self.preview_text.delete(1.0, tk.END)
            return
            
        # Clear preview
        self.preview_text.delete(1.0, tk.END)
        
        # Format data for preview
        data = self.current_anomaly.data
        
        # Convert temperatures for display if needed
        max_temp = data['max_temp']
        mean_temp = data['mean_temp']
        delta_t = data['delta_t']
        
        if self.use_fahrenheit:
            max_temp = self.celsius_to_fahrenheit(max_temp)
            mean_temp = self.celsius_to_fahrenheit(mean_temp)
            delta_t = delta_t * 9/5  # Convert delta T
            
        # Unit for display
        unit = "°F" if self.use_fahrenheit else "°C"
        
        # Special preview for sun reflections
        if data['is_sun_reflection'] or data['defect_type'] == 'Sun reflection':
            preview = (
                f"ID: {data['id']}\n"
                f"Type: {data['defect_type']}\n"
                f"Component: {data['component_type']}\n"
                f"Severity: N/A (Sun Reflection)\n"
                f"Pattern: {data['pattern']}\n"
                f"Area: {data['area']:.1f} px\n"
                f"Max Temp: {max_temp:.1f}{unit}\n"
                f"Mean Temp: {mean_temp:.1f}{unit}\n"
                f"Aspect Ratio: {data['aspect_ratio']:.2f}\n"
                f"Power Loss: 0.0 W (No Impact)\n"
                f"Note: Sun reflections are marked for exclusion\n"
                f"      from thermal analysis to avoid false positives.\n"
            )
        else:
            preview = (
                f"ID: {data['id']}\n"
                f"Type: {data['defect_type']}\n"
                f"Component: {data['component_type']}\n"
                f"Severity: {data['severity']}\n"
                f"Pattern: {data['pattern']}\n"
                f"Delta T: {delta_t:.1f}{unit}\n"
                f"Area: {data['area']:.1f} px\n"
                f"Max Temp: {max_temp:.1f}{unit}\n"
                f"Mean Temp: {mean_temp:.1f}{unit}\n"
                f"Aspect Ratio: {data['aspect_ratio']:.2f}\n"
                f"Power Loss: {data['power_loss_kw']*1000:.1f} W\n"
            )
        
        self.preview_text.insert(tk.END, preview)
        
    def export_data(self):
        # Ensure we save the current image's annotations
        self.save_current_annotations()
        
        # Check if we have any annotations
        if not self.image_annotations:
            messagebox.showinfo("Info", "No anomalies to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Training Data",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            # Collect all annotations
            all_data = []
            
            for image_path, annotations in self.image_annotations.items():
                image_name = os.path.basename(image_path)
                
                for anomaly_data in annotations:
                    # Create a copy of the anomaly data
                    row = anomaly_data.copy()
                    
                    # Add image filename
                    row['image_filename'] = image_name
                    
                    # Convert centroid to separate x,y columns
                    x, y = row.pop('centroid')
                    row['centroid_x'] = x
                    row['centroid_y'] = y
                    
                    # Convert bbox if exists
                    if 'bbox' in row:
                        x, y, w, h = row.pop('bbox')
                        row['bbox_x'] = x
                        row['bbox_y'] = y
                        row['bbox_width'] = w
                        row['bbox_height'] = h
                    
                    # Convert temperatures if needed
                    if self.use_fahrenheit:
                        row['max_temp_f'] = self.celsius_to_fahrenheit(row['max_temp'])
                        row['mean_temp_f'] = self.celsius_to_fahrenheit(row['mean_temp'])
                        row['delta_t_f'] = row['delta_t'] * 9/5
                    else:
                        row['max_temp_c'] = row['max_temp']
                        row['mean_temp_c'] = row['mean_temp']
                        row['delta_t_c'] = row['delta_t']
                    
                    all_data.append(row)
                    
            # Create dataframe
            df = pd.DataFrame(all_data)
            
            # Save based on extension
            if file_path.lower().endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
                
            print(f"Exported {len(all_data)} anomalies from {len(self.image_annotations)} images to {file_path}")
            messagebox.showinfo("Export Complete", f"Exported {len(all_data)} anomalies from {len(self.image_annotations)} images to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
            print(f"Error exporting data: {str(e)}")
        
if __name__ == "__main__":
    app = TrainingDataGenerator()
    app.mainloop()