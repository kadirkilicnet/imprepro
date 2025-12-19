import sys
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, 
                               QGroupBox, QScrollArea, QSplitter, QFileDialog, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QCheckBox, QFrame, QSizePolicy, QLineEdit)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QObject, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QAction, QIcon, QWheelEvent, QPainter, QColor, QPen
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem

from processors import FilterRegistry
from color_adjust import ColorAdjuster
from utils import resize_image, get_image_stats

class WorkerSignals(QObject):
    finished = Signal(object, object, object, float) # processed_image, geo_image, stats, duration
    error = Signal(str)

class ImageWorker(QThread):
    def __init__(self, original_image, filter_stack, color_params, geometric_params):
        super().__init__()
        self.original_image = original_image
        self.filter_stack = filter_stack # List of (name, params)
        self.color_params = color_params
        self.geometric_params = geometric_params
        self.signals = WorkerSignals()

    def run(self):
        try:
            t0 = time.time()
            img = self.original_image.copy()

            # 1. Geometric Operations
            from utils import rotate_image, crop_image, flip_image, apply_affine, apply_perspective
            
            # Order: Crop -> Rotate -> Flip -> Affine -> Perspective -> Resize
            geo = self.geometric_params
            
            if geo.get('crop_w', 0) > 0 and geo.get('crop_h', 0) > 0:
                img = crop_image(img, geo['crop_x'], geo['crop_y'], geo['crop_w'], geo['crop_h'])
                
            if geo.get('rotate', 0) != 0:
                img = rotate_image(img, geo['rotate'])
                
            flip = geo.get('flip') # 'None', 'Horizontal', 'Vertical', 'Both'
            if flip == 'Horizontal': img = flip_image(img, 1)
            elif flip == 'Vertical': img = flip_image(img, 0)
            elif flip == 'Both': img = flip_image(img, -1)
            

            target_w = geo.get('width')
            target_h = geo.get('height')
            interp = geo.get('interp', cv2.INTER_LINEAR)
            
            h, w = img.shape[:2]
            if (target_w and target_w != w) or (target_h and target_h != h):
                 img = resize_image(img, width=target_w, height=target_h, method=interp)

            # Capture state after geometric ops but before color/filters
            geo_image = img.copy()

            # 2. Color Adjust
            img = ColorAdjuster.process(
                img, 
                brightness=self.color_params['brightness'],
                contrast=self.color_params['contrast'],
                saturation=self.color_params['saturation'],
                hue=self.color_params['hue'],
                gamma=self.color_params['gamma'],
                clahe_limit=self.color_params.get('clahe', 0.0)
            )

            # 3. Filter Stack
            for filter_name, filter_params in self.filter_stack:
                if not filter_name: continue
                processor = FilterRegistry.get_filter(filter_name)
                if processor:
                    res = processor.apply(img, filter_params)
                    if len(res.shape) == 2 and len(img.shape) == 3:
                         res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
                    img = res

            t1 = time.time()
            duration = (t1 - t0) * 1000
            
            stats = get_image_stats(img)
            self.signals.finished.emit(img, geo_image, stats, duration)

        except Exception as e:
            self.signals.error.emit(str(e))

class HandleItem(QGraphicsRectItem):
    def __init__(self, pos_index, parent):
        super().__init__(0, 0, 10, 10, parent)
        self.pos_index = pos_index # 0:TL, 1:TR, 2:BR, 3:BL
        self.pos_index = pos_index # 0:TL, 1:TR, 2:BR, 3:BL
        self.setBrush(QColor(255, 255, 0))
        self.setZValue(10) # Ensure handles are above the rect
        self.setFlag(QGraphicsRectItem.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges)

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionChange:
            self.parentItem().on_handle_move(self.pos_index, value)
        return super().itemChange(change, value)

class ResizableSelectionRect(QGraphicsRectItem):
    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.setFlag(QGraphicsRectItem.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges)
        self.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        self.handle_size = 10
        self.handles = []
        for i in range(4):
            h = HandleItem(i, self)
            self.handles.append(h)
        self._is_updating = False
        self.update_handles()

    def update_handles(self):
        if self._is_updating: return
        self._is_updating = True
        r = self.rect()
        self.handles[0].setPos(r.left() - self.handle_size/2, r.top() - self.handle_size/2)
        self.handles[1].setPos(r.right() - self.handle_size/2, r.top() - self.handle_size/2)
        self.handles[2].setPos(r.right() - self.handle_size/2, r.bottom() - self.handle_size/2)
        self.handles[3].setPos(r.left() - self.handle_size/2, r.bottom() - self.handle_size/2)
        self._is_updating = False

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionChange or change == QGraphicsRectItem.ItemScaleChange:
            # Emit signal for sync (using scene bounding rect)
            pass
        return super().itemChange(change, value)
    
    def on_handle_move(self, index, pos):
        if self._is_updating: return
        self._is_updating = True
        r = self.rect()
        if index == 0: r.setTopLeft(pos + QPointF(self.handle_size/2, self.handle_size/2))
        elif index == 1: r.setTopRight(pos + QPointF(self.handle_size/2, self.handle_size/2))
        elif index == 2: r.setBottomRight(pos + QPointF(self.handle_size/2, self.handle_size/2))
        elif index == 3: r.setBottomLeft(pos + QPointF(self.handle_size/2, self.handle_size/2))
        self.setRect(r.normalized())
        self.update_handles()
        self._is_updating = False
        # Emit selection_changed via parent view if possible, but let's do it simply
        if self.scene() is None: return
        views = self.scene().views()
        if not views: return
        view = views[0]
        if hasattr(view, 'selection_changed') and not view._syncing:
            rect = self.sceneBoundingRect()
            view.selection_changed.emit(rect.x(), rect.y(), rect.width(), rect.height())

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.update_handles()
        if self.scene() is None: return
        views = self.scene().views()
        if not views: return
        view = views[0]
        if hasattr(view, 'selection_changed') and not view._syncing:
            rect = self.sceneBoundingRect()
            view.selection_changed.emit(rect.x(), rect.y(), rect.width(), rect.height())

class ImageGraphicsView(QGraphicsView):
    """Custom QGraphicsView with zoom and synchronization capabilities."""
    zoom_changed = Signal(float, float, float) # Scale factor, relative center X, relative center Y
    pan_changed = Signal(int, int) # Horizontal, Vertical scroll values
    selection_changed = Signal(float, float, float, float) # x, y, w, h

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        print("ImageGraphicsView: Pixmap item added")
        
        self.selection_rect = ResizableSelectionRect(QRectF(10, 10, 100, 100))
        self.scene.addItem(self.selection_rect)
        self.selection_rect.hide()
        
        self.current_zoom = 1.0
        self._syncing = False

        self.horizontalScrollBar().valueChanged.connect(self._on_h_scroll)
        self.verticalScrollBar().valueChanged.connect(self._on_v_scroll)
        
    def show_selection(self, show):
        self.selection_rect.setVisible(show)

    def get_selection(self):
        # Map scene selection rect to the pixmap item coordinates
        r_scene = self.selection_rect.sceneBoundingRect()
        r_item = self.pixmap_item.mapFromScene(r_scene).boundingRect()
        return r_item.x(), r_item.y(), r_item.width(), r_item.height()

    def set_selection(self, x, y, w, h):
        self._syncing = True
        self.selection_rect.setPos(x, y)
        self.selection_rect.setRect(0, 0, w, h)
        self.selection_rect.update_handles()
        self._syncing = False

    def set_image(self, cv_img, reset_view=True):
        if cv_img is None: return
        
        height, width = cv_img.shape[:2]
        if len(cv_img.shape) == 2:
            q_img = QImage(cv_img.data, width, height, width, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, width, height, 3*width, QImage.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(0, 0, width, height)
        
        # Initial fit only if requested
        if reset_view:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.current_zoom = self.transform().m11()
            # Also reset selection rect if new image loaded
            self.selection_rect.setRect(10, 10, 100, 100)
            self.selection_rect.update_handles()

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        scale_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor

        self.scale(scale_factor, scale_factor)
        self.current_zoom *= scale_factor
        
        # Emit signal for synchronization
        # We pass dummy logic here, the connection in MainWindow handles absolute sync
        self.zoom_changed.emit(scale_factor, 0.5, 0.5) 
        # Also sync scroll bars immediately
        self.pan_changed.emit(self.horizontalScrollBar().value(), self.verticalScrollBar().value())

    def _on_h_scroll(self, val):
        if not self._syncing:
            self.pan_changed.emit(val, self.verticalScrollBar().value())

    def _on_v_scroll(self, val):
        if not self._syncing:
            self.pan_changed.emit(self.horizontalScrollBar().value(), val)

    def get_view_state(self):
        """Returns the absolute view state (transform, h_val, v_val)."""
        return self.transform(), self.horizontalScrollBar().value(), self.verticalScrollBar().value()

    def set_view_state(self, state):
        """Sets the absolute view state."""
        if self._syncing: return
        self._syncing = True
        transform, h, v = state
        self.setTransform(transform)
        self.horizontalScrollBar().setValue(h)
        self.verticalScrollBar().setValue(v)
        self.current_zoom = transform.m11()
        self._syncing = False

    def sync_transform(self, scale_factor):
        # Deprecated for absolute sync but kept for relative fallback
        pass

    def sync_scroll(self, h, v):
        # Deprecated for absolute sync
        pass

class FilterItemWidget(QFrame):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        
        header = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(["None"] + FilterRegistry.get_filter_names())
        self.combo.currentTextChanged.connect(self.on_filter_changed)
        
        btn_remove = QPushButton("X")
        btn_remove.setFixedWidth(30)
        btn_remove.clicked.connect(lambda: self.main_window.remove_filter_item(self))
        
        header.addWidget(self.combo)
        header.addWidget(btn_remove)
        layout.addLayout(header)
        
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.params_container)
        
        self.filter_inputs = {}

    def on_filter_changed(self, name):
        for i in reversed(range(self.params_layout.count())): 
            self.params_layout.itemAt(i).widget().setParent(None)
        self.filter_inputs = {}
        
        if name == "None":
            self.main_window.trigger_update()
            return

        processor = FilterRegistry.get_filter(name)
        if not processor: return

        params = processor.get_params()
        for p in params:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0,0,0,0)
            row_layout.addWidget(QLabel(p['label']))
            
            widget = None
            if p['type'] == 'int':
                widget = QSpinBox()
                widget.setRange(p.get('min', 0), p.get('max', 999))
                widget.setValue(p['default'])
                widget.valueChanged.connect(self.main_window.trigger_update)
            elif p['type'] == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(p.get('min', 0.0), p.get('max', 999.0))
                widget.setValue(p['default'])
                widget.setSingleStep(p.get('step', 0.1))
                widget.valueChanged.connect(self.main_window.trigger_update)
            elif p['type'] == 'option':
                widget = QComboBox()
                widget.addItems(p['options'])
                widget.setCurrentText(p['default'])
                widget.currentIndexChanged.connect(self.main_window.trigger_update)
            elif p['type'] == 'text':
                widget = QLineEdit(str(p['default']))
                widget.editingFinished.connect(self.main_window.trigger_update)
            
            if widget:
                self.filter_inputs[p['name']] = widget
                row_layout.addWidget(widget)
            if p['type'] != 'matrix':
                self.params_layout.addWidget(row)
        
        if hasattr(self.main_window, 'trigger_update'):
            self.main_window.trigger_update()

    def get_params(self):
        name = self.combo.currentText()
        if name == "None": return None, {}
        
        params = {}
        for p_name, widget in self.filter_inputs.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                params[p_name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[p_name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                params[p_name] = widget.text()
            elif isinstance(widget, MatrixInputWidget):
                params[p_name] = widget.get_matrix_string()
        return name, params


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("imprepro")
        from utils import resource_path
        self.setWindowIcon(QIcon(resource_path("logo.png"))) # Set App Icon
        self.resize(1200, 800)
        self.resize(1200, 800)
        
        # State
        self.original_image = None
        self.processed_image = None
        self.current_geo_image = None # Image after geometric transforms
        self.current_worker = None
        
        # Undo/Redo State
        self.history = []
        self.redo_stack = []
        self.is_undoing = False
        
        # Debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(200) 
        self.update_timer.timeout.connect(self.start_processing)
        
        self.init_ui()
        self.setup_sync()
        self.setup_shortcuts()

    def setup_shortcuts(self):
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo)
        self.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self.redo)
        self.addAction(self.redo_action)

    def setup_sync(self):
        # Synchronize views only if toggle is on
        self.lbl_original.selection_changed.connect(self.on_selection_sync)
        self.lbl_processed.selection_changed.connect(self.on_selection_sync)
        
        # Remove old sync connections since we define them in init_ui now or here
        # Actually let's just clear and redefine to be safe or leave them if we removed the slots in the class
        pass

    def on_selection_sync(self, x, y, w, h):
        if self.check_sync_view.isChecked():
            sender = self.sender()
            target = self.lbl_processed if sender == self.lbl_original else self.lbl_original
            target.set_selection(x, y, w, h)

    def on_zoom_sync(self, s, x, y):
        if self.check_sync_view.isChecked():
            sender = self.sender()
            target = self.lbl_processed if sender == self.lbl_original else self.lbl_original
            target.sync_transform(s)

    def on_pan_sync(self, h, v):
        if self.check_sync_view.isChecked():
            sender = self.sender()
            target = self.lbl_processed if sender == self.lbl_original else self.lbl_original
            target.sync_scroll(h, v)

    def init_ui(self):
        print("Entering init_ui")
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        print("Main layout created")

        # Pre-initialize labels so they can be referenced in Sidebar controls
        self.lbl_original = ImageGraphicsView()
        self.lbl_processed = ImageGraphicsView()
        
        # Connect signals for Absolute Sync
        def sync_aboslute(source, target):
             if self.check_sync_view.isChecked():
                 target.set_view_state(source.get_view_state())

        self.lbl_original.zoom_changed.connect(lambda: sync_aboslute(self.lbl_original, self.lbl_processed))
        self.lbl_original.pan_changed.connect(lambda: sync_aboslute(self.lbl_original, self.lbl_processed))
        
        self.lbl_processed.zoom_changed.connect(lambda: sync_aboslute(self.lbl_processed, self.lbl_original))
        self.lbl_processed.pan_changed.connect(lambda: sync_aboslute(self.lbl_processed, self.lbl_original))
        
        print("Labels created")
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.lbl_original)
        self.splitter.addWidget(self.lbl_processed)

        # --- Left Sidebar (Controls) ---
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        # sidebar.setFixedWidth(320) # Removed fixed width to allow resizing
        sidebar.setMinimumWidth(320) # Set minimum instead
        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setAlignment(Qt.AlignTop)
        print("Sidebar scaffolding created")

        # 1. File Group
        file_group = QGroupBox("File Operations")
        file_layout = QGridLayout()
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        btn_save = QPushButton("Save Result")
        btn_save.clicked.connect(self.save_image)
        btn_import = QPushButton("Import Settings")
        btn_import.clicked.connect(self.import_settings)
        btn_export = QPushButton("Export Settings")
        btn_export.clicked.connect(self.export_settings)
        
        file_layout.addWidget(btn_load, 0, 0)
        file_layout.addWidget(btn_save, 0, 1)
        file_layout.addWidget(btn_import, 1, 0)
        file_layout.addWidget(btn_export, 1, 1)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self.redo)
        file_layout.addWidget(self.btn_undo, 2, 0)
        file_layout.addWidget(self.btn_redo, 2, 1)

        file_group.setLayout(file_layout)
        sidebar_layout.addWidget(file_group)
        print("File group added")

        # 2. Geometric Operations Group
        resize_group = QGroupBox("Geometric Operations")
        resize_layout = QGridLayout()
        
        self.spin_width = QSpinBox()
        self.spin_width.setRange(0, 10000)
        self.spin_width.setSpecialValueText("Original")
        self.spin_height = QSpinBox()
        self.spin_height.setRange(0, 10000)
        self.spin_height.setSpecialValueText("Original")
        
        self.check_aspect = QCheckBox("Keep Aspect Ratio")
        self.check_aspect.setChecked(True)
        
        # Connect resize triggers
        self.spin_width.valueChanged.connect(self.on_width_changed)
        self.spin_height.valueChanged.connect(self.on_height_changed)
        
        # Interpolation Combo
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(["Nearest", "Linear", "Cubic", "Area"])
        self.combo_interp.setCurrentIndex(1)
        self.combo_interp.currentIndexChanged.connect(self.trigger_update)

        resize_layout.addWidget(QLabel("W:"), 0, 0)
        resize_layout.addWidget(self.spin_width, 0, 1)
        resize_layout.addWidget(QLabel("H:"), 0, 2)
        resize_layout.addWidget(self.spin_height, 0, 3)
        resize_layout.addWidget(self.check_aspect, 1, 0, 1, 2)
        
        self.btn_reset_size = QPushButton("Reset to Original")
        self.btn_reset_size.clicked.connect(self.reset_size)
        resize_layout.addWidget(self.btn_reset_size, 1, 2, 1, 2)

        resize_layout.addWidget(QLabel("Method:"), 2, 0)
        resize_layout.addWidget(self.combo_interp, 2, 1, 1, 3)
        
        # New Geometric Ops
        resize_layout.addWidget(QFrame(), 3, 0, 1, 4) # Spacer
        
        self.check_show_selection = QCheckBox("Show Crop Selection")
        self.check_show_selection.toggled.connect(self.lbl_original.show_selection)
        self.check_show_selection.toggled.connect(self.lbl_processed.show_selection)
        resize_layout.addWidget(self.check_show_selection, 4, 0, 1, 2)
        
        self.btn_apply_crop = QPushButton("Crop to Selection")
        self.btn_apply_crop.clicked.connect(self.apply_selection_crop)
        resize_layout.addWidget(self.btn_apply_crop, 4, 2, 1, 2)
        
        resize_layout.addWidget(QLabel("Rotate:"), 5, 0)
        self.spin_rotate = QDoubleSpinBox(); self.spin_rotate.setRange(-360, 360)
        resize_layout.addWidget(self.spin_rotate, 5, 1, 1, 3)
        
        resize_layout.addWidget(QLabel("Flip:"), 6, 0)
        self.combo_flip = QComboBox()
        self.combo_flip.addItems(["None", "Horizontal", "Vertical", "Both"])
        resize_layout.addWidget(self.combo_flip, 6, 1, 1, 3)
        

        # Hidden spinboxes for crop params (updated from selection)
        self.spin_crop_x = QSpinBox(); self.spin_crop_x.hide()
        self.spin_crop_y = QSpinBox(); self.spin_crop_y.hide()
        self.spin_crop_w = QSpinBox(); self.spin_crop_w.hide()
        self.spin_crop_h = QSpinBox(); self.spin_crop_h.hide()
        
        # Connect signals
        self.spin_rotate.valueChanged.connect(self.trigger_update)
        self.combo_flip.currentIndexChanged.connect(self.trigger_update)
        
        resize_group.setLayout(resize_layout)
        sidebar_layout.addWidget(resize_group)
        resize_group.setLayout(resize_layout)
        sidebar_layout.addWidget(resize_group)

        self._in_resize_update = False

        # 3. Color Group
        color_group = QGroupBox("Color Adjustments")
        color_layout = QGridLayout() # Label, Slider, Value
        
        self.color_controls = {} # store pointers to get values

        def add_slider(name, min_val, max_val, default, row, scale=1.0):
            lbl = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(min_val * scale), int(max_val * scale))
            slider.setValue(int(default * scale))
            val_lbl = QLabel(str(default))
            
            btn_reset = QPushButton("R")
            btn_reset.setFixedWidth(25)
            btn_reset.setToolTip(f"Reset {name}")
            btn_reset.clicked.connect(lambda: slider.setValue(int(default * scale)))
            
            slider.valueChanged.connect(lambda v: val_lbl.setText(str(round(v/scale, 2))))
            slider.valueChanged.connect(self.trigger_update)
            
            color_layout.addWidget(lbl, row, 0)
            color_layout.addWidget(slider, row, 1)
            color_layout.addWidget(val_lbl, row, 2)
            color_layout.addWidget(btn_reset, row, 3)
            
            self.color_controls[name] = {'slider': slider, 'scale': scale, 'default': default}

        add_slider("Brightness", -255, 255, 0, 0)
        add_slider("Contrast", 0.1, 3.0, 1.0, 1, 100.0) 
        add_slider("Saturation", 0.0, 3.0, 1.0, 2, 100.0)
        add_slider("Hue", -180, 180, 0, 3)
        add_slider("Gamma", 0.1, 3.0, 1.0, 4, 100.0)
        add_slider("CLAHE", 0.0, 10.0, 0.0, 5, 10.0)
        
        btn_reset_color = QPushButton("Reset All Colors")
        btn_reset_color.clicked.connect(self.reset_colors)
        color_layout.addWidget(btn_reset_color, 6, 0, 1, 4)

        color_group.setLayout(color_layout)
        sidebar_layout.addWidget(color_group)

        # 4. Filter Group
        filter_group = QGroupBox("Filters and Operations")
        filter_layout = QVBoxLayout()
        
        self.btn_add_filter = QPushButton("+ Add Filter")
        self.btn_add_filter.clicked.connect(self.add_filter_item)
        filter_layout.addWidget(self.btn_add_filter)
        
        self.filter_stack_widget = QWidget()
        self.filter_stack_layout = QVBoxLayout(self.filter_stack_widget)
        self.filter_stack_layout.setContentsMargins(0,0,0,0)
        
        self.filter_items = []

        filter_layout.addWidget(self.filter_stack_widget)
        filter_group.setLayout(filter_layout)
        sidebar_layout.addWidget(filter_group)

        # 5. View Group
        view_group = QGroupBox("View Settings")
        view_layout = QVBoxLayout()
        self.check_show_original = QCheckBox("Show Original Image")
        self.check_show_original.setChecked(True)
        self.check_show_original.toggled.connect(self.on_show_original_toggled)
        
        self.check_sync_view = QCheckBox("Synchronize Zoom/Pan")
        self.check_sync_view.setChecked(True)
        
        view_layout.addWidget(self.check_show_original)
        view_layout.addWidget(self.check_sync_view)
        
        self.check_show_analytics = QCheckBox("Show Analytics Panel")
        self.check_show_analytics.setChecked(False)
        self.check_show_analytics.toggled.connect(self.on_show_analytics_toggled)
        view_layout.addWidget(self.check_show_analytics)
        
        view_group.setLayout(view_layout)
        sidebar_layout.addWidget(view_group)

        sidebar.setWidget(sidebar_content)
        main_layout.addWidget(sidebar)

        main_layout.addWidget(self.splitter, stretch=1)

        # --- Analytics ---
        self.analytics_group = QGroupBox("Analytics")
        self.analytics_group.setFixedWidth(250)
        analytics_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        analytics_layout.addWidget(self.stats_table)
        self.analytics_group.setLayout(analytics_layout)
        main_layout.addWidget(self.analytics_group)
        self.analytics_group.hide() # Hidden by default

        self.status_lbl = QLabel("Ready")
        self.statusBar().addWidget(self.status_lbl)
        print("Exiting init_ui")

    def on_show_original_toggled(self, checked):
        if checked:
            self.lbl_original.show()
        else:
            self.lbl_original.hide()

    def on_show_analytics_toggled(self, checked):
        if checked:
            self.analytics_group.show()
        else:
            self.analytics_group.hide()

    def reset_size(self):
        self._in_resize_update = True
        self.spin_width.setValue(0)
        self.spin_height.setValue(0)
        self._in_resize_update = False
        self.trigger_update()

    def on_width_changed(self, w):
        if self._in_resize_update or self.original_image is None or not self.check_aspect.isChecked():
            self.trigger_update()
            return
        if w == 0: return
        self._in_resize_update = True
        h_orig, w_orig = self.original_image.shape[:2]
        new_h = int(w * h_orig / w_orig)
        self.spin_height.setValue(new_h)
        self._in_resize_update = False
        self.trigger_update()

    def on_height_changed(self, h):
        if self._in_resize_update or self.original_image is None or not self.check_aspect.isChecked():
            self.trigger_update()
            return
        if h == 0: return
        self._in_resize_update = True
        h_orig, w_orig = self.original_image.shape[:2]
        new_w = int(h * w_orig / h_orig)
        self.spin_width.setValue(new_w)
        self._in_resize_update = False
        self.trigger_update()

    def add_filter_item(self):
        item = FilterItemWidget(self)
        self.filter_items.append(item)
        self.filter_stack_layout.addWidget(item)
        self.trigger_update()

    def remove_filter_item(self, item):
        self.filter_items.remove(item)
        item.setParent(None)
        self.trigger_update()

    def export_settings(self):
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Export Settings", "settings.json", "JSON (*.json)")
        if path:
            c_params, f_stack, r_params = self.get_current_params()
            data = {
                "adjustments": c_params,
                "filters": f_stack,
                "resize": r_params,
                "analytics": self.current_stats if hasattr(self, 'current_stats') else {}
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            self.status_lbl.setText(f"Settings exported to {path}")

    def trigger_update(self):
        self.update_timer.start()

    def reset_colors(self):
        for name, data in self.color_controls.items():
            data['slider'].setValue(int(data['default'] * data['scale']))
        self.trigger_update()

    def import_settings(self):
        import json
        path, _ = QFileDialog.getOpenFileName(self, "Import Settings", "", "JSON (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Apply Adjustments
                adj = data.get('adjustments', {})
                for name, val in adj.items():
                    # Map lowercase names back to Title case if needed
                    for ctrl_name, ctrl_data in self.color_controls.items():
                        if ctrl_name.lower() == name:
                            ctrl_data['slider'].setValue(int(val * ctrl_data['scale']))
                
                # Apply Geometric Ops
                # SKIPPED PER USER REQUEST: Keep current image shape/size/rotation exactly as is.
                # Use import as "Style Transfer" only.
                '''
                geo = data.get('geometric', {}) or data.get('resize', {})
                
                # Import Logic - robust for different image sizes
                # Only apply Color, Filters, and "Safe" geometry (Flip, Rotate, Interp)
                # Ignore crop and resize as they depend on pixel dimensions
                safe_geo = geo.copy()
                for unsafe_key in ['width', 'height', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 'affine', 'perspective']:
                    if unsafe_key in safe_geo: del safe_geo[unsafe_key]
                
                # Update Geometry (Partial)
                self.spin_rotate.setValue(safe_geo.get('rotate') or 0)
                self.combo_flip.setCurrentText(safe_geo.get('flip') or "None")
                
                interp = safe_geo.get('interp', cv2.INTER_LINEAR)
                methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
                if interp in methods:
                    self.combo_interp.setCurrentIndex(methods.index(interp))
                '''
                
                # Apply Filters
                filters = data.get('filters', [])
                for item in self.filter_items:
                    item.setParent(None)
                self.filter_items = []
                
                for f_name, f_params in filters:
                    item = FilterItemWidget(self)
                    self.filter_items.append(item)
                    self.filter_stack_layout.addWidget(item)
                    item.combo.setCurrentText(f_name)
                    # Trigger on_filter_changed manually if needed, 
                    # but setCurrentText should have triggered it.
                    for p_name, p_val in f_params.items():
                        if p_name in item.filter_inputs:
                            w = item.filter_inputs[p_name]
                            if isinstance(w, (QSpinBox, QDoubleSpinBox)): w.setValue(p_val)
                            elif isinstance(w, QComboBox): w.setCurrentText(str(p_val))
                            elif isinstance(w, QLineEdit): w.setText(str(p_val))

                self.trigger_update()
                self.status_lbl.setText(f"Settings imported from {path}")
            except Exception as e:
                self.status_lbl.setText(f"Error importing settings: {e}")

    def save_state(self, force=False):
        if self.is_undoing: return
        c, f, g = self.get_current_params()
        # Store copy of current original image to support undo of crops
        img_state = self.original_image.copy() if self.original_image is not None else None
        state = {'color': c, 'filters': f, 'geometric': g, 'image': img_state}
        
        if not self.history:
            self.history.append(state)
            self.redo_stack = []
            return

        last_state = self.history[-1]
        
        # Helper to check equality without crashing on numpy arrays
        def states_equal(s1, s2):
            if s1.keys() != s2.keys(): return False
            # Check basic params
            if s1['color'] != s2['color']: return False
            if s1['filters'] != s2['filters']: return False
            if s1['geometric'] != s2['geometric']: return False
            
            # Check image existence mismatch
            img1 = s1.get('image')
            img2 = s2.get('image')
            if (img1 is None) != (img2 is None): return False
            
            # If both have images, check if they are the same object reference or shape/content
            # Optimization: If we are here, we might assume they are equal if same shape?
            # Or just assume if force=True we skip this.
            # Safe bet: If 'force' is True, we save.
            # If not force, we check params. If params same, we assume image same 
            # (unless explicit crop action triggered force save).
            return True

        if force or not states_equal(last_state, state):
            self.history.append(state)
            if len(self.history) > 50: self.history.pop(0)
            self.redo_stack = []

    def undo(self):
        if len(self.history) <= 1: return
        self.redo_stack.append(self.history.pop())
        state = self.history[-1]
        self.apply_state(state)

    def redo(self):
        if not self.redo_stack: return
        state = self.redo_stack.pop()
        self.history.append(state)
        self.apply_state(state)

    def apply_state(self, state):
        self.is_undoing = True
        
        # Restore Image State if present
        if 'image' in state and state['image'] is not None:
             self.original_image = state['image'].copy()
             self.lbl_original.set_image(self.original_image, reset_view=False)
        
        # Color
        for name, val in state['color'].items():
            for ctrl_name, ctrl_data in self.color_controls.items():
                if ctrl_name.lower() == name:
                    ctrl_data['slider'].setValue(int(val * ctrl_data['scale']))
        
        # Geometric
        geo = state['geometric']
        self._in_resize_update = True
        self.spin_width.setValue(geo.get('width') or 0)
        self.spin_height.setValue(geo.get('height') or 0)
        self.spin_crop_x.setValue(geo.get('crop_x') or 0)
        self.spin_crop_y.setValue(geo.get('crop_y') or 0)
        self.spin_crop_w.setValue(geo.get('crop_w') or 0)
        self.spin_crop_h.setValue(geo.get('crop_h') or 0)
        self.spin_rotate.setValue(geo.get('rotate') or 0)
        self.combo_flip.setCurrentText(geo.get('flip') or "None")
        self.edit_affine.setText(geo.get('affine') or "")
        self.edit_persp.setText(geo.get('perspective') or "")
        
        interp = geo.get('interp', cv2.INTER_LINEAR)
        methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        if interp in methods:
            self.combo_interp.setCurrentIndex(methods.index(interp))
        self._in_resize_update = False
        
        # Filters
        for item in self.filter_items: item.setParent(None)
        self.filter_items = []
        for f_name, f_params in state['filters']:
            item = FilterItemWidget(self)
            self.filter_items.append(item)
            self.filter_stack_layout.addWidget(item)
            item.combo.setCurrentText(f_name)
            for p_name, p_val in f_params.items():
                if p_name in item.filter_inputs:
                    w = item.filter_inputs[p_name]
                    if isinstance(w, (QSpinBox, QDoubleSpinBox)): w.setValue(p_val)
                    elif isinstance(w, QComboBox): w.setCurrentText(str(p_val))
                    elif isinstance(w, QLineEdit): w.setText(str(p_val))
        
        self.is_undoing = False
        self.trigger_update()

    def get_current_params(self):
        c_params = {}
        for name, data in self.color_controls.items():
            val = data['slider'].value() / data['scale']
            c_params[name.lower()] = val
            
        f_stack = []
        for item in self.filter_items:
            name, params = item.get_params()
            if name:
                f_stack.append((name, params))
                
        w = self.spin_width.value() or None
        h = self.spin_height.value() or None
        interp_idx = self.combo_interp.currentIndex()
        methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        
        geo_params = {
            'width': w, 'height': h, 'interp': methods[interp_idx],
            'crop_x': self.spin_crop_x.value(), 'crop_y': self.spin_crop_y.value(),
            'crop_w': self.spin_crop_w.value(), 'crop_h': self.spin_crop_h.value(),
            'rotate': self.spin_rotate.value(),
            'flip': self.combo_flip.currentText(),
            'flip': self.combo_flip.currentText()
        }
        return c_params, f_stack, geo_params

    def start_processing(self):
        if self.original_image is None: return
        self.save_state()
        c_params, f_stack, geo_params = self.get_current_params()
        self.current_worker = ImageWorker(self.original_image, f_stack, c_params, geo_params)
        self.current_worker.signals.finished.connect(self.on_processing_finished)
        self.current_worker.signals.error.connect(lambda e: self.status_lbl.setText(f"Error: {e}"))
        self.current_worker.start()
        self.status_lbl.setText("Processing...")

    def on_processing_finished(self, img, geo_img, stats, duration):
        self.processed_image = img
        self.current_geo_image = geo_img
        self.current_stats = stats
        
        # Update Processed View
        self.lbl_processed.set_image(img, reset_view=False)
        
        # Update Original View to show the geometric transformed version
        # This keeps the two views aligned in size and rotation
        self.lbl_original.set_image(geo_img, reset_view=False)
        
        # FORCE Absolute Sync from Original -> Processed to match exact state
        if self.check_sync_view.isChecked():
             self.lbl_processed.set_view_state(self.lbl_original.get_view_state())

        self.status_lbl.setText(f"Processed in {duration:.1f} ms")
        self.update_stats_table(stats)

    def update_stats_table(self, stats):
        self.stats_table.setRowCount(0)
        for k, v in stats.items():
            row = self.stats_table.rowCount()
            self.stats_table.insertRow(row)
            self.stats_table.setItem(row, 0, QTableWidgetItem(str(k)))
            val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
            self.stats_table.setItem(row, 1, QTableWidgetItem(val_str))

    def apply_selection_crop(self):
        if self.current_geo_image is None: return
        
        # Save state BEFORE modifying the image
        self.save_state(force=True)
        
        try:
            # Get selection in item coordinates
            r_scene = self.lbl_original.selection_rect.sceneBoundingRect()
            # Map from scene to the displayed pixmap item (which now holds current_geo_image)
            r_item = self.lbl_original.pixmap_item.mapFromScene(r_scene).boundingRect()
            
            x = int(r_item.x())
            y = int(r_item.y())
            w = int(r_item.width())
            h = int(r_item.height())
            
            # Using current_geo_image because that's what the user is seeing and selecting on
            target_img = self.current_geo_image
            ih, iw = target_img.shape[:2]
            
            # Clamp to image bounds
            x = max(0, min(x, iw-1))
            y = max(0, min(y, ih-1))
            w = max(1, min(w, iw-x))
            h = max(1, min(h, ih-y))
            
            # Commit Crop: 
            # We are cropping the "Geometrically Transformed" image.
            # This means the new "original" will be this cropped result.
            # Since this result already has Rotation/Flip/Resize applied, 
            # we MUST reset those parameters to avoid double-application.
            
            cropped = target_img[y:y+h, x:x+w].copy()
            self.original_image = cropped
            
            # Since we baked the transforms into the new original, reset the UI controls
            self.spin_crop_x.setValue(0)
            self.spin_crop_y.setValue(0)
            self.spin_crop_w.setValue(0)
            self.spin_crop_h.setValue(0)
            
            self.spin_width.setValue(0)
            self.spin_height.setValue(0) 
            self.spin_rotate.setValue(0)
            self.combo_flip.setCurrentIndex(0) # None
            
            # Reset view to fit the new cropped image properly
            self.lbl_original.set_image(self.original_image, reset_view=True)
            self.lbl_processed.set_image(self.original_image, reset_view=True) # Temp until process finishes
            
            self.check_show_selection.setChecked(False)
            self.lbl_original.selection_rect.hide()
            
            self.trigger_update()
        except Exception as e:
            print(f"Crop error: {e}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            stream = open(path, "rb")
            bytes_data = bytearray(stream.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            if img is not None:
                self.original_image = img
                self.lbl_original.set_image(img, reset_view=True)
                self._in_resize_update = True
                self.spin_width.setValue(0) 
                self.spin_height.setValue(0)
                self._in_resize_update = False
                self.trigger_update()

    def save_image(self):
        if self.processed_image is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "output.png", "Images (*.png *.jpg)")
        if path:
            cv2.imwrite(path, self.processed_image)
