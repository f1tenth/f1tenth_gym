import sys
import time
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QSurfaceFormat

fmt = QSurfaceFormat()
fmt.setSwapInterval(0)  # 0 = no vsync, 1 = vsync
QSurfaceFormat.setDefaultFormat(fmt)


class OpenGLRectangleWithFPS(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Filled Rectangle with FPS")
        # Set up GL view widget
        self.view = gl.GLViewWidget()
        self.setCentralWidget(self.view)

        # Fix camera to XY view
        self.view.opts['elevation'] = 90  # Top-down
        self.view.opts['azimuth'] = 0     # Facing +Y
        self.view.setCameraPosition(distance=40)

        # 🔧 Replace mouse drag behavior: allow pan (right drag), disable rotation
        self._enable_pan_only()

        # Add 2D background using GLImageItem
        map_img = np.zeros((200, 300), dtype=np.uint8)
        map_img[50:150, 100:200] = 255
        map_rgb = np.stack([map_img]*3, axis=-1)
        alpha = np.ones((map_rgb.shape[0], map_rgb.shape[1], 1), dtype=np.uint8) * 255
        map_rgba = np.concatenate((map_rgb, alpha), axis=-1)
        map_rgba = np.flip(np.rot90(map_rgba, k=1), axis=0)
        image_item = gl.GLImageItem(map_rgba)
        image_item.translate(0, 0, -0.01)
        image_item.scale(0.1, 0.1, 1)
        # self.view.addItem(image_item)

        # Define rectangle corners (XY plane, z=0)
        rectangle = np.array([
            [0, 0, 0],
            [5, 0, 0],
            [5, 3, 0],
            [0, 3, 0]
        ], dtype=np.float32)

        # Line outline
        # outline = np.vstack([rectangle, rectangle[0]])
        # rect_item = gl.GLLinePlotItem(pos=outline, color=(1, 0, 0, 1), width=2, mode='line_strip')
        # self.view.addItem(rect_item)

        # Fill with GLMeshItem
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        mesh_item = gl.GLMeshItem(
            vertexes=rectangle,
            faces=faces,
            faceColors=np.array([[0, 0.7, 0.9, 1.0], [0, 0.7, 0.9, 1.0]]),
            smooth=False,
            drawEdges=False
        )
        self.view.addItem(mesh_item)

        # FPS label
        self.fps_label = QtWidgets.QLabel(self)
        font = QtGui.QFont("Courier New", 14, QtGui.QFont.Weight.Bold)
        self.fps_label.setFont(font)
        self.fps_label.setStyleSheet("color: white; background-color: black; padding: 2px;")
        self.fps_label.move(10, 10)
        self.fps_label.resize(100, 20)
        self.fps_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Frame timer
        self.last_time = time.time()
        self.frame_count = 0
        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self._update_fps)
        # self.timer.start(0)

    def _enable_pan_only(self):
        """Override GLViewWidget events to disable rotation and allow right-click panning."""
        self.view.pan_active = False
        self.view.pan_start = QtCore.QPoint()

        def mousePressEvent(event):
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.view.pan_active = True
                self.view.pan_start = event.pos()
                event.accept()
            else:
                event.ignore()

        def mouseMoveEvent(event):
            if self.view.pan_active:
                delta = event.pos() - self.view.pan_start
                dx = -delta.y() * 0.01
                dy = -delta.x() * 0.01
                self.view.pan(dx, dy, 0)
                self.view.pan_start = event.pos()
                event.accept()
            else:
                event.ignore()

        def mouseReleaseEvent(event):
            self.view.pan_active = False
            event.accept()

        self.view.mousePressEvent = mousePressEvent
        self.view.mouseMoveEvent = mouseMoveEvent
        self.view.mouseReleaseEvent = mouseReleaseEvent

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time

        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.last_time = now
            self.frame_count = 0
        self.view.update()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = OpenGLRectangleWithFPS()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
