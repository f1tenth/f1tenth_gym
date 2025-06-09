import sys
import time
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl


class OpenGLRectangleWithFPS(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Rectangle with FPS")

        # Set up GL view widget
        self.view = gl.GLViewWidget()
        self.setCentralWidget(self.view)
        self.view.setCameraPosition(distance=20, elevation=90, azimuth=0)

        # Rectangle
        rectangle = np.array([
            [0, 0, 0],
            [5, 0, 0],
            [5, 3, 0],
            [0, 3, 0],
            [0, 0, 0]
        ], dtype=np.float32)

        rect_item = gl.GLLinePlotItem(pos=rectangle, color=(1, 0, 0, 1), width=2, mode='line_strip')
        self.view.addItem(rect_item)

        # FPS label (parent is the main window, not GL view!)
        self.fps_label = QtWidgets.QLabel(self)
        self.fps_label.setText("FPS: ...")
        self.fps_label.setStyleSheet("color: white; background-color: rgba(0,0,0,180); padding: 4px;")
        self.fps_label.setFont(QtGui.QFont("Courier", 14, QtGui.QFont.Weight.Bold))
        self.fps_label.setGeometry(10, 10, 150, 30)
        self.fps_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.fps_label.raise_()  # Ensure it's on top

        # Timing
        self.last_time = time.time()
        self.frame_count = 0

        # Timer for render loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.render)
        self.timer.start(0)

    def render(self):
        self._update_fps()
        self.view.update()

    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.last_time = current_time
            self.frame_count = 0


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = OpenGLRectangleWithFPS()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
