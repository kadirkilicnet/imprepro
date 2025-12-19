import sys
from PySide6.QtWidgets import QApplication
from ui import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Set a dark theme or style if desired
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

