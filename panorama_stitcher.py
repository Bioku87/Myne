import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QFileDialog, QProgressBar, QCheckBox, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import gc  # For memory management
try:
    import piexif  # For EXIF metadata extraction
    from PIL import Image  # For image handling with EXIF
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False
    print("Warning: piexif or PIL not found - geo positioning will be disabled")
    
try:
    import utm  # For UTM conversion
    UTM_AVAILABLE = True
except ImportError:
    UTM_AVAILABLE = False
    print("Warning: utm package not found - coordinate conversion will be disabled")

class PanoramaStitcher:
    """Class for stitching multiple thermal drone images into a panorama with visible grid layout."""
    
    def __init__(self, use_gpu=False, layout_mode="grid"):
        self.use_gpu = use_gpu
        self.layout_mode = layout_mode  # "grid", "auto", or "geo"
        
        # Create OpenCV stitcher with appropriate settings
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        
        # Configure stitcher for better results with thermal images
        try:
            # Try to set advanced parameters if available in OpenCV version
            self.stitcher.setRegistrationResol(0.6)  # Higher accuracy in registration
            self.stitcher.setPanoConfidenceThresh(0.3)  # Lower threshold for better matches
            self.stitcher.setWaveCorrection(True)  # Enable wave correction
            self.stitcher.setWaveCorrectKind(cv2.detail.WAVE_CORRECT_HORIZ)  # Horizontal correction
            
            # Set blender to multi-band by default
            blender = cv2.detail.MultiBandBlender()
            self.stitcher.setBlender(blender)
        except Exception as e:
            print(f"Warning: Could not set advanced stitcher parameters: {str(e)}")
        
        if use_gpu:
            try:
                self.stitcher.setFeaturesFinder(cv2.cuda_GpuFeaturesFinder())
                self.stitcher.setWarper(cv2.cuda_WarperGpu())
            except Exception as e:
                print(f"GPU acceleration not available: {str(e)}")
                self.use_gpu = False
        
        # Store intermediate results
        self.feature_matches = []
        self.stitch_progress = 0
        self.panorama = None
        self.stitch_map = None
        self.downsample_factor = 1  # No downsampling by default
        self.geo_transforms = []  # Store geo transforms for visualization
        
    def stitch_images(self, images, image_paths=None, callback=None, downsample_factor=None):
        """Stitch a list of images into a panorama.
        
        Args:
            images: List of images to stitch
            image_paths: Optional list of paths to the images (for extracting geo data)
            callback: Progress callback function(progress, message)
            downsample_factor: Optional factor to downsample images for faster stitching
            
        Returns:
            Stitched panorama image
        """
        if len(images) < 2:
            if len(images) == 1:
                return images[0]
            return None
        
        # Apply downsampling if requested
        if downsample_factor and downsample_factor > 1:
            downsampled_images = []
            for img in images:
                h, w = img.shape[:2]
                downsampled = cv2.resize(img, (w//downsample_factor, h//downsample_factor))
                downsampled_images.append(downsampled)
            processed_images = downsampled_images
        else:
            processed_images = images.copy()
        
        # Convert all images to similar format
        for i in range(len(processed_images)):
            # Ensure image is in correct format
            if len(processed_images[i].shape) == 2:
                # Convert grayscale to RGB for stitching
                processed_images[i] = cv2.cvtColor(processed_images[i], cv2.COLOR_GRAY2RGB)
            
            # Normalize and convert if needed
            if processed_images[i].dtype != np.uint8:
                # Normalize to 0-255 range
                processed_images[i] = cv2.normalize(processed_images[i], None, 0, 255, cv2.NORM_MINMAX)
                processed_images[i] = processed_images[i].astype(np.uint8)
            
            # Update progress
            if callback:
                progress = (i / len(processed_images)) * 20  # Up to 20%
                callback(progress, f"Preprocessing image {i+1}/{len(processed_images)}")
        
        # Choose stitching method based on layout_mode
        if self.layout_mode == "auto":
            # Try OpenCV's built-in stitcher first
            if callback:
                callback(20, "Attempting automatic stitching...")
            
            panorama = self._auto_stitch(processed_images, callback)
            
            # If automatic stitching fails, fall back to grid layout
            if panorama is None:
                if callback:
                    callback(40, "Automatic stitching failed, switching to grid layout...")
                
                panorama = self._grid_stitch(processed_images, callback)
        
        elif self.layout_mode == "geo" and image_paths and (EXIF_AVAILABLE and UTM_AVAILABLE):
            if callback:
                callback(20, "Using geo-based positioning...")
            
            # Extract and use geo data for positioning
            panorama = self._geo_stitch(processed_images, image_paths, callback)
            
            # If geo stitching fails, fall back to grid layout
            if panorama is None:
                if callback:
                    callback(40, "Geo-based stitching failed, switching to grid layout...")
                
                panorama = self._grid_stitch(processed_images, callback)
        
        else:
            # Use grid layout as the default option
            if callback:
                callback(20, "Creating grid layout panorama...")
            
            panorama = self._grid_stitch(processed_images, callback)
        
        if panorama is not None:
            self.panorama = panorama
            
            # Create stitch map showing image placement
            if callback:
                callback(90, "Creating stitch map")
            
            self.stitch_map = self._create_stitch_map(processed_images, panorama)
            
            if callback:
                callback(100, "Stitching complete")
        else:
            if callback:
                callback(0, "Stitching failed completely")
        
        return panorama
    
    def _auto_stitch(self, images, callback=None):
        """Attempt automatic stitching using OpenCV's built-in stitcher."""
        try:
            if callback:
                callback(25, "Running feature detection...")
            
            # Try OpenCV's built-in stitcher
            status, result = self.stitcher.stitch(images)
            
            if status == cv2.Stitcher_OK:
                if callback:
                    callback(50, "Automatic stitching successful")
                return result
            else:
                # Try with custom pairwise stitching
                if callback:
                    callback(30, "Default stitcher failed, trying pairwise stitching...")
                
                result = self._stitch_pairwise(images, callback)
                
                if result is not None:
                    # Check if the resulting image is mostly empty
                    # This happens when images are incorrectly aligned
                    non_black_ratio = np.sum((result != 0).any(axis=2)) / (result.shape[0] * result.shape[1])
                    
                    if non_black_ratio < 0.3:  # If less than 30% of the image has content
                        if callback:
                            callback(40, "Poor stitch quality detected, will try grid layout")
                        return None
                    
                    return result
        except Exception as e:
            print(f"Error in auto stitching: {str(e)}")
        
        return None
    
    def _grid_stitch(self, images, callback=None):
        """Stitch images in a grid layout to ensure all images are visible."""
        if not images:
            return None
        
        if callback:
            callback(50, "Creating grid layout...")
        
        # Determine grid dimensions
        n_images = len(images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        rows = grid_size
        cols = grid_size
        
        # Optional: optimize grid aspect ratio
        if grid_size * (grid_size - 1) >= n_images:
            rows = grid_size - 1
        
        # Get max dimensions for uniformity
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        
        # Add borders to make all images the same size
        bordered_images = []
        for img in images:
            h, w = img.shape[:2]
            
            # Create border to make image uniform size
            top = (max_height - h) // 2
            bottom = max_height - h - top
            left = (max_width - w) // 2
            right = max_width - w - left
            
            # Add border (black)
            bordered = cv2.copyMakeBorder(
                img, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            
            bordered_images.append(bordered)
        
        # Create the grid
        total_height = rows * max_height
        total_width = cols * max_width
        
        # Create canvas with 10% margin between images
        margin = int(min(max_height, max_width) * 0.05)  # 5% margin
        
        canvas_height = rows * max_height + (rows - 1) * margin
        canvas_width = cols * max_width + (cols - 1) * margin
        
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place images on the grid
        for i, img in enumerate(bordered_images):
            if i >= n_images:
                break
                
            row = i // cols
            col = i % cols
            
            y_offset = row * (max_height + margin)
            x_offset = col * (max_width + margin)
            
            # Place image
            panorama[
                y_offset:y_offset + max_height,
                x_offset:x_offset + max_width
            ] = img
            
            if callback:
                progress = 50 + (i / n_images) * 40
                callback(progress, f"Placing image {i+1}/{n_images} on grid")
        
        return panorama
    
    def _geo_stitch(self, images, image_paths, callback=None):
        """Position images based on geolocation data."""
        if not (EXIF_AVAILABLE and UTM_AVAILABLE):
            return None
            
        if callback:
            callback(30, "Extracting geo data...")
        
        # Extract geo data
        geo_data = self._extract_geo_data(image_paths)
        
        # Check if we have valid geo data
        if not geo_data or all(g['latitude'] is None for g in geo_data):
            return None
        
        if callback:
            callback(40, "Computing geo transformations...")
        
        # Compute transforms
        transforms = self._compute_geo_transforms(geo_data)
        
        if not transforms:
            return None
            
        self.geo_transforms = transforms  # Store for visualization
        
        if callback:
            callback(50, "Applying geo transformations...")
        
        # Apply transforms
        transformed_images = self._apply_geo_transforms(images, image_paths, transforms)
        
        if callback:
            callback(70, "Creating composite image...")
        
        # Create a composite image
        panorama = self._create_geo_composite(transformed_images)
        
        return panorama
    
    def _extract_geo_data(self, image_paths):
        """Extract geolocation data from image EXIF metadata."""
        if not EXIF_AVAILABLE:
            return None
            
        geo_data = []
        
        for path in image_paths:
            try:
                # Open image and extract EXIF data
                img = Image.open(path)
                
                # Check if EXIF data exists
                if 'exif' not in img.info:
                    geo_data.append({
                        'path': path,
                        'latitude': None,
                        'longitude': None,
                        'altitude': None,
                        'direction': None,
                        'width': img.width,
                        'height': img.height
                    })
                    continue
                    
                exif_dict = piexif.load(img.info['exif'])
                
                # Extract GPS info
                gps_info = exif_dict.get('GPS', {})
                
                if gps_info:
                    # Extract latitude
                    lat_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef, b'N')
                    lat = gps_info.get(piexif.GPSIFD.GPSLatitude)
                    
                    # Extract longitude
                    lon_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef, b'E')
                    lon = gps_info.get(piexif.GPSIFD.GPSLongitude)
                    
                    # Extract altitude
                    alt_ref = gps_info.get(piexif.GPSIFD.GPSAltitudeRef, 0)
                    alt = gps_info.get(piexif.GPSIFD.GPSAltitude)
                    
                    # Extract heading (direction)
                    heading = gps_info.get(piexif.GPSIFD.GPSImgDirection)
                    heading_ref = gps_info.get(piexif.GPSIFD.GPSImgDirectionRef)
                    
                    # Convert to decimal degrees
                    latitude = self._convert_to_decimal(lat, lat_ref)
                    longitude = self._convert_to_decimal(lon, lon_ref)
                    
                    # Convert altitude to meters
                    altitude = None
                    if alt:
                        altitude = alt[0] / alt[1]
                        if alt_ref == 1:  # Below sea level
                            altitude = -altitude
                    
                    # Convert heading to degrees
                    direction = None
                    if heading:
                        direction = heading[0] / heading[1]
                    
                    # Get image dimensions
                    width, height = img.size
                    
                    geo_data.append({
                        'path': path,
                        'latitude': latitude,
                        'longitude': longitude,
                        'altitude': altitude,
                        'direction': direction,
                        'width': width,
                        'height': height
                    })
                else:
                    # No GPS data
                    geo_data.append({
                        'path': path,
                        'latitude': None,
                        'longitude': None,
                        'altitude': None,
                        'direction': None,
                        'width': img.width,
                        'height': img.height
                    })
            except Exception as e:
                print(f"Error extracting geo data from {path}: {str(e)}")
                # Add placeholder entry
                geo_data.append({
                    'path': path,
                    'latitude': None,
                    'longitude': None,
                    'altitude': None,
                    'direction': None,
                    'width': 0,
                    'height': 0
                })
        
        return geo_data
    
    def _convert_to_decimal(self, dms, ref):
        """Convert GPS coordinates from DMS to decimal format."""
        if dms is None:
            return None
            
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1] / 60.0
        seconds = dms[2][0] / dms[2][1] / 3600.0
        
        decimal = degrees + minutes + seconds
        
        # If southern or western hemisphere, negate
        if ref in (b'S', b'W'):
            decimal = -decimal
            
        return decimal
    
    def _compute_geo_transforms(self, geo_data):
        """Compute transformations based on geolocation data."""
        if not UTM_AVAILABLE:
            return None
            
        transforms = []
        
        # Filter out entries with missing geo data
        valid_geo = [g for g in geo_data if g['latitude'] is not None]
        
        if not valid_geo:
            return None
            
        # Convert lat/lon to UTM coordinates
        utm_coords = []
        for g in valid_geo:
            try:
                utm_x, utm_y, zone_number, zone_letter = utm.from_latlon(g['latitude'], g['longitude'])
                utm_coords.append({
                    'path': g['path'],
                    'utm_x': utm_x,
                    'utm_y': utm_y,
                    'altitude': g['altitude'],
                    'direction': g['direction'],
                    'width': g['width'],
                    'height': g['height'],
                    'lat': g['latitude'],
                    'lon': g['longitude']
                })
            except Exception as e:
                print(f"Error converting to UTM: {str(e)}")
        
        if not utm_coords:
            return None
            
        # Compute min/max coordinates to normalize
        min_x = min(u['utm_x'] for u in utm_coords)
        min_y = min(u['utm_y'] for u in utm_coords)
        
        # Compute transforms
        for i, coord in enumerate(utm_coords):
            # Normalize coordinates (scale to pixel space)
            # Scale factor determined by image density
            scale_factor = 10  # pixels per meter - adjust as needed
            
            x = (coord['utm_x'] - min_x) * scale_factor
            y = (coord['utm_y'] - min_y) * scale_factor
            
            # Create transformation matrix for translation
            T = np.array([
                [1, 0, x],
                [0, 1, y],
                [0, 0, 1]
            ])
            
            # Apply rotation if available
            if coord['direction'] is not None:
                # Convert heading to radians (North = 0°)
                theta = np.radians(coord['direction'])
                
                # Center of rotation (image center)
                center_x = coord['width'] / 2
                center_y = coord['height'] / 2
                
                # Translation to origin
                T1 = np.array([
                    [1, 0, -center_x],
                    [0, 1, -center_y],
                    [0, 0, 1]
                ])
                
                # Rotation matrix
                R = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                
                # Translation back
                T2 = np.array([
                    [1, 0, center_x],
                    [0, 1, center_y],
                    [0, 0, 1]
                ])
                
                # Combine matrices
                T = T @ T2 @ R @ T1
            
            transforms.append({
                'path': coord['path'],
                'transform': T,
                'utm_x': coord['utm_x'],
                'utm_y': coord['utm_y'],
                'lat': coord['lat'],
                'lon': coord['lon']
            })
        
        return transforms
    
    def _apply_geo_transforms(self, images, image_paths, transforms):
        """Apply geometric transformations based on geo data."""
        if not transforms:
            return images
            
        transformed_images = []
        transform_lookup = {t['path']: t['transform'] for t in transforms}
        
        # Determine output size
        max_x = max_y = 0
        
        for i, path in enumerate(image_paths):
            if path in transform_lookup:
                # Get original image dimensions
                h, w = images[i].shape[:2]
                
                # Get corners of the image
                corners = np.array([
                    [0, 0, 1],
                    [w, 0, 1],
                    [w, h, 1],
                    [0, h, 1]
                ])
                
                # Apply transformation to corners
                transform = transform_lookup[path]
                transformed_corners = []
                for corner in corners:
                    transformed_corner = transform @ corner
                    transformed_corners.append([
                        transformed_corner[0] / transformed_corner[2],
                        transformed_corner[1] / transformed_corner[2]
                    ])
                
                # Update max dimensions
                for x, y in transformed_corners:
                    max_x = max(max_x, int(np.ceil(x)))
                    max_y = max(max_y, int(np.ceil(y)))
        
        # Add padding to ensure all image content is captured
        max_width = int(max_x * 1.1)  # 10% padding
        max_height = int(max_y * 1.1)  # 10% padding
        
        # Apply transformations
        for i, path in enumerate(image_paths):
            if path in transform_lookup:
                # Get transformation matrix
                transform = transform_lookup[path]
                
                # Warp the image
                warped = cv2.warpPerspective(
                    images[i],
                    transform,
                    (max_width, max_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_TRANSPARENT
                )
                
                transformed_images.append(warped)
            else:
                # If no transform, just add original
                transformed_images.append(images[i])
        
        return transformed_images
    
    def _create_geo_composite(self, images):
        """Create a composite from transformed images."""
        if not images:
            return None
        
        # Find dimensions of the composite
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        
        # Create empty canvas
        composite = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        
        # Add each image to the canvas
        for img in images:
            # Create mask for non-black pixels
            mask = (img != 0).any(axis=2)
            
            # Add image to composite where mask is True
            composite[mask] = img[mask]
        
        return composite
    
    def _stitch_pairwise(self, images, callback=None):
        """Stitch images pairwise."""
        if len(images) < 2:
            return images[0] if images else None
        
        # Start with first image
        result = images[0]
        
        # Stitch each subsequent image
        for i in range(1, len(images)):
            if callback:
                progress = 30 + (i / (len(images) - 1)) * 20
                callback(progress, f"Stitching image {i+1}/{len(images)}")
            
            # Try OpenCV stitcher
            try:
                status, pair_result = self.stitcher.stitch([result, images[i]])
                
                if status == cv2.Stitcher_OK:
                    result = pair_result
                    continue
            except Exception:
                pass
            
            # If OpenCV stitcher fails, try custom method
            try:
                # Find matches between images
                matches = self._find_matches(result, images[i])
                
                if matches and len(matches) >= 4:
                    # Find homography
                    H, _ = cv2.findHomography(matches[:, 1], matches[:, 0], cv2.RANSAC)
                    
                    # Calculate output dimensions
                    h1, w1 = result.shape[:2]
                    h2, w2 = images[i].shape[:2]
                    
                    # Calculate corners of the second image after transformation
                    corners = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
                    corners_transformed = cv2.perspectiveTransform(corners, H)
                    
                    # Find minimum x,y values
                    [x_min, y_min] = np.int32(corners_transformed.min(axis=0)[0])
                    [x_max, y_max] = np.int32(corners_transformed.max(axis=0)[0])
                    
                    # Calculate output size
                    x_min = min(0, x_min)
                    y_min = min(0, y_min)
                    output_width = max(w1, x_max) - x_min
                    output_height = max(h1, y_max) - y_min
                    
                    # Create translation matrix
                    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
                    H_translated = translation @ H
                    
                    # Warp second image
                    warped = cv2.warpPerspective(
                        images[i], 
                        H_translated, 
                        (output_width, output_height)
                    )
                    
                    # Place first image on the canvas
                    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    canvas[-y_min:h1-y_min, -x_min:w1-x_min] = result
                    
                    # Create mask for warped image
                    mask = (warped != 0).any(axis=2)
                    
                    # Create mask for first image
                    first_mask = np.zeros((output_height, output_width), dtype=bool)
                    first_mask[-y_min:h1-y_min, -x_min:w1-x_min] = (result != 0).any(axis=2)
                    
                    # Calculate overlap region
                    overlap = mask & first_mask
                    
                    # Place second image where there's no overlap
                    mask_no_overlap = mask & ~overlap
                    canvas[mask_no_overlap] = warped[mask_no_overlap]
                    
                    # Blend in overlap region
                    if np.any(overlap):
                        # Simple 50-50 blend in overlap region
                        canvas[overlap] = (0.5 * canvas[overlap] + 0.5 * warped[overlap]).astype(np.uint8)
                    
                    result = canvas
                else:
                    # Not enough matches, place images side by side
                    h1, w1 = result.shape[:2]
                    h2, w2 = images[i].shape[:2]
                    
                    # Create canvas to place both images
                    max_h = max(h1, h2)
                    canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
                    
                    # Place images side by side
                    canvas[0:h1, 0:w1] = result
                    canvas[0:h2, w1:w1+w2] = images[i]
                    
                    result = canvas
            except Exception as e:
                print(f"Error in pairwise stitching: {str(e)}")
                
                # Place images side by side if all other methods fail
                h1, w1 = result.shape[:2]
                h2, w2 = images[i].shape[:2]
                
                max_h = max(h1, h2)
                canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
                
                canvas[0:h1, 0:w1] = result
                canvas[0:h2, w1:w1+w2] = images[i]
                
                result = canvas
        
        return result
    
    def _find_matches(self, img1, img2):
        """Find feature matches between two images."""
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Try ORB features first
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
        
        # If not enough features, try SIFT-like
        if (descriptors1 is None or descriptors2 is None or 
            len(keypoints1) < 10 or len(keypoints2) < 10):
            # Try AKAZE or other detectors
            try:
                detector = cv2.AKAZE_create()
                keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
                keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
            except:
                # Fall back to BRISK if AKAZE not available
                try:
                    detector = cv2.BRISK_create()
                    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
                    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
                except:
                    return None
        
        # Match descriptors
        if descriptors1 is not None and descriptors2 is not None:
            if descriptors1.dtype != np.uint8:
                # For float descriptors (like SIFT/SURF)
                matcher = cv2.BFMatcher(cv2.NORM_L2)
            else:
                # For binary descriptors (like ORB/BRISK)
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Match features
            matches = matcher.match(descriptors1, descriptors2)
            
            # Sort them by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Take only good matches
            good_matches = matches[:min(50, len(matches))]
            
            # Extract matched point coordinates
            points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
            points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
            
            # Store for visualization
            self.feature_matches.append((good_matches, keypoints1, keypoints2))
            
            # Return points for homography
            return np.column_stack((points1, points2))
        
        return None
    
    def _create_stitch_map(self, images, panorama):
        """Create a visualization of how images were stitched."""
        if panorama is None:
            return None
        
        # Create a copy for visualization
        stitch_map = panorama.copy()
        
        # For grid layout, highlight all image borders
        for i, img in enumerate(images):
            # Determine color for this image
            color = [
                (255, 0, 0),   # Red
                (0, 255, 0),   # Green
                (0, 0, 255),   # Blue
                (255, 255, 0), # Yellow
                (255, 0, 255), # Magenta
                (0, 255, 255), # Cyan
                (255, 128, 0), # Orange
                (128, 0, 255)  # Purple
            ][i % 8]
            
            if self.layout_mode == "grid":
                # For grid layout, borders are predictable
                h, w = img.shape[:2]
                max_h = max(img.shape[0] for img in images)
                max_w = max(img.shape[1] for img in images)
                
                margin = int(min(max_h, max_w) * 0.05)  # 5% margin
                cols = int(np.ceil(np.sqrt(len(images))))
                
                # Calculate position
                row = i // cols
                col = i % cols
                
                y_offset = row * (max_h + margin)
                x_offset = col * (max_w + margin)
                
                # Draw rectangle
                cv2.rectangle(
                    stitch_map, 
                    (x_offset, y_offset), 
                    (x_offset + max_w - 1, y_offset + max_h - 1), 
                    color, 
                    3
                )
                
                # Add image number
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    stitch_map, 
                    f"{i+1}", 
                    (x_offset + 10, y_offset + 30), 
                    font, 
                    1, 
                    color, 
                    2
                )
            else:
                # For auto or geo layouts, boundaries are irregular
                # Just add a label at a reasonable position
                h_pano, w_pano = panorama.shape[:2]
                
                # Place label at 1/4 points for illustration 
                # (this is approximate since we don't know exact positions)
                rows = int(np.ceil(np.sqrt(len(images))))
                cols = int(np.ceil(len(images) / rows))
                
                # Calculate approximate positions
                row = i // cols
                col = i % cols
                
                center_x = int((col + 0.5) * w_pano / cols)
                center_y = int((row + 0.5) * h_pano / rows)
                
                # Add label
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    stitch_map, 
                    f"{i+1}", 
                    (center_x, center_y), 
                    font, 
                    2, 
                    color, 
                    3
                )
        
        return stitch_map
    
    def get_panorama(self):
        """Get the stitched panorama."""
        return self.panorama
    
    def get_stitch_map(self):
        """Get the visualization of how images were stitched."""
        return self.stitch_map


class StitchingThread(QThread):
    """Worker thread for image stitching."""
    progress_updated = pyqtSignal(int, str)
    stitch_updated = pyqtSignal(np.ndarray)
    stitching_complete = pyqtSignal(np.ndarray, np.ndarray)  # panorama, stitch_map
    error_occurred = pyqtSignal(str)
    
    def __init__(self, images, image_paths=None, layout_mode="grid", downsample_factor=2):
        super().__init__()
        self.images = images
        self.image_paths = image_paths
        self.stitcher = PanoramaStitcher(layout_mode=layout_mode)
        self.downsample_factor = downsample_factor
    
    def run(self):
        try:
            def progress_callback(progress, message):
                self.progress_updated.emit(progress, message)
                if self.stitcher.panorama is not None:
                    self.stitch_updated.emit(self.stitcher.panorama)
            
            # Stitch images
            panorama = self.stitcher.stitch_images(
                self.images,
                self.image_paths,
                progress_callback, 
                downsample_factor=self.downsample_factor
            )
            
            if panorama is not None:
                stitch_map = self.stitcher.get_stitch_map()
                self.stitching_complete.emit(panorama, stitch_map)
            else:
                self.error_occurred.emit("Stitching failed")
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_occurred.emit(f"Error: {str(e)}\n{error_details}")


class PanoramaApp(QMainWindow):
    """Main application for thermal image stitching."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SolarEye Thermal Panorama Creator")
        self.setMinimumSize(1000, 800)
        
        self.images = []
        self.image_paths = []
        self.panorama = None
        self.stitch_map = None
        self.layout_mode = "grid"  # Default to grid layout
        
        self._create_ui()
    
    def _create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Top title
        title_layout = QHBoxLayout()
        title_label = QLabel("SolarEye Thermal Panorama Creator")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Main controls
        control_layout = QHBoxLayout()
        
        load_button = QPushButton("Load Images")
        load_button.setMinimumWidth(120)
        load_button.setStyleSheet("font-weight: bold;")
        load_button.clicked.connect(self.load_images)
        control_layout.addWidget(load_button)
        
        # Layout mode selection
        layout_label = QLabel("Layout Mode:")
        control_layout.addWidget(layout_label)
        
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Grid Layout (Always Visible)")
        self.layout_combo.addItem("Auto Stitch (Feature-Based)")
        if EXIF_AVAILABLE and UTM_AVAILABLE:
            self.layout_combo.addItem("Geo Layout (GPS-Based)")
        control_layout.addWidget(self.layout_combo)
        
        stitch_button = QPushButton("Create Panorama")
        stitch_button.setMinimumWidth(120)
        stitch_button.setStyleSheet("font-weight: bold;")
        stitch_button.clicked.connect(self.stitch_images)
        control_layout.addWidget(stitch_button)
        
        save_button = QPushButton("Save Panorama")
        save_button.setMinimumWidth(120)
        save_button.clicked.connect(self.save_panorama)
        control_layout.addWidget(save_button)
        
        layout.addLayout(control_layout)
        
        # Status
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        status_layout.addWidget(self.progress_bar)
        
        layout.addLayout(status_layout)
        
        # Image display
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Load images to begin')
        self.ax.axis('off')
        self.canvas.draw()
    
    def load_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", 
            "Image Files (*.jpg *.jpeg *.png *.tif *.tiff);;All Files (*.*)"
        )
        
        if not file_paths:
            return
        
        self.images = []
        self.image_paths = []
        
        # Load images
        for path in file_paths:
            img = cv2.imread(path)
            if img is not None:
                # Convert BGR to RGB for display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img_rgb)
                self.image_paths.append(path)
        
        # Display loaded images
        self.display_loaded_images()
        
        self.status_label.setText(f"Loaded {len(self.images)} images")
    
    def display_loaded_images(self):
        """Display loaded images in a grid."""
        if not self.images:
            return
        
        # Clear plot
        self.figure.clear()
        
        # Determine grid size
        n_images = len(self.images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        # Create grid of subplots
        for i, img in enumerate(self.images):
            if i < cols * rows:
                ax = self.figure.add_subplot(rows, cols, i+1)
                ax.imshow(img)
                ax.set_title(f"Image {i+1}")
                ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def stitch_images(self):
        """Stitch loaded images."""
        if len(self.images) < 2:
            self.status_label.setText("Need at least 2 images to stitch")
            return
        
        # Memory management - force garbage collection
        gc.collect()
        
        # Get layout mode
        layout_mode_text = self.layout_combo.currentText()
        if "Grid" in layout_mode_text:
            self.layout_mode = "grid"
        elif "Auto" in layout_mode_text:
            self.layout_mode = "auto"
        else:
            self.layout_mode = "geo"
        
        # Start stitching
        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing images...")
        
        # Determine if downsampling is needed
        downsample_factor = 1
        if len(self.images) > 10:
            # For large sets, use more aggressive downsampling
            downsample_factor = 3
        elif len(self.images) > 5:
            downsample_factor = 2
        
        # Create worker thread
        self.worker = StitchingThread(
            self.images, 
            self.image_paths, 
            self.layout_mode, 
            downsample_factor
        )
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.stitch_updated.connect(self.update_stitch_display)
        self.worker.stitching_complete.connect(self.stitching_complete)
        self.worker.error_occurred.connect(self.stitching_error)
        
        # Start worker
        self.worker.start()
    
    def update_progress(self, progress, message):
        """Update progress bar and status."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def update_stitch_display(self, panorama):
        """Update display with intermediate stitching result."""
        # Clear plot
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Display panorama
        self.ax.imshow(panorama)
        self.ax.set_title("Stitching in progress...")
        self.ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def stitching_complete(self, panorama, stitch_map):
        """Handle completed stitching."""
        self.panorama = panorama
        self.stitch_map = stitch_map
        
        # Clear plot
        self.figure.clear()
        
        # Create tabs using subplots
        ax1 = self.figure.add_subplot(121)
        ax1.imshow(panorama)
        ax1.set_title("Panorama")
        ax1.axis('off')
        
        ax2 = self.figure.add_subplot(122)
        ax2.imshow(stitch_map)
        ax2.set_title("Image Positions")
        ax2.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        self.status_label.setText("Panorama creation complete")
        self.progress_bar.setValue(100)
        
        # Force garbage collection
        gc.collect()
    
    def stitching_error(self, error_msg):
        """Handle stitching error."""
        self.status_label.setText(f"Error: {error_msg}")
        self.progress_bar.setValue(0)
        
        # Show error details in a dialog
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Stitching Error", error_msg)
    
    def save_panorama(self):
        """Save stitched panorama."""
        if self.panorama is None:
            self.status_label.setText("No panorama to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Panorama", "", 
            "Image Files (*.jpg *.png *.tif);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Convert RGB to BGR for OpenCV
            panorama_bgr = cv2.cvtColor(self.panorama, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, panorama_bgr)
            self.status_label.setText(f"Saved panorama to {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error saving: {str(e)}")


# Run application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PanoramaApp()
    window.show()
    sys.exit(app.exec_())
