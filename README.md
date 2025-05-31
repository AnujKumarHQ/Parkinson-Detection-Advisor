# Parkinson-Detection-Advisor
A Python-based GUI tool for predicting Parkinson's disease using vocal features from the UCI dataset.it features a Tkinter interface with manual input and batch prediction modes, leveraging a Random Forest model. Key functionalities include confusion matrix and feature importance visualizations, LaTeX PDF reports, and a Treatment tab with an expanded recommendation box (~80% of tab, ~10% margins) offering lifestyle and medical advice. Built with pandas, numpy, scikit-learn, matplotlib, and seaborn, it supports Pyodide for browser execution. Ideal for researchers and developers exploring AI-driven medical diagnostics. Open-source, with a medical disclaimer: consult professionals for diagnosis. Contributions welcome!

Features:
- Manual Input: Train a Random Forest model and predict for a single patient.
- Pre-trained Model: Analyze uploaded CSV data for batch predictions.
- Visualizations: Confusion matrix and feature importance plots.
- PDF Reports: LaTeX-based reports with predictions, metrics, and advice.
- Treatment Tab: Lifestyle, diet, and medical recommendations for Parkinson's.
- Export Results: Save predictions to the GUI console.
- Terms: Explanations of medical/technical terms.
- Pyodide-compatible for browser execution.

Dataset:
- Source: https://archive.ics.uci.edu/ml/datasets/parkinsons
- Features: 22 vocal attributes (e.g., MDVP:Fo(Hz), Jitter, Shimmer, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE)
- Target: Status (1 = Parkinson's, 0 = Healthy)

Prerequisites
-------------
- Python 3.8+ (https://www.python.org/downloads/)
- Tkinter (included with Python)
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- LaTeX (for PDF reports):
  - MikTeX (https://miktex.org/download)
  - latexmk (install via MikTeX Console)
- VS Code (recommended, https://code.visualstudio.com/)
- Internet connection (for UCI dataset download)
- Windows 10/11

Setup
-----
1. Install Python 3.8+:
   - Download and install from https://www.python.org/downloads/
   - Verify: Open Command Prompt, run `python --version`

2. Install libraries:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Install MikTeX (for PDF reports):
   - Download from https://miktex.org/download
   - Install and open MikTeX Console
   - Update packages and install `latexmk`
   - Verify: `latexmk --version`

4. Set up VS Code:
   - Install VS Code and Python extension
   - Open folder: `File -> Open Folder -> d:\Parkison`
   - Select Python interpreter: `Ctrl+Shift+P -> Python: Select Interpreter`

5. Save project files:
   - Copy `parkinsons_gui_no_database.py` to `d:\Parkison`
   - Ensure no conflicting older files (e.g., `parkinsons_gui_enhanced_with_terms.py`)

Running
-------
1. Open VS Code, navigate to `d:\Parkison`
2. Open `parkinsons_gui_no_database.py`
3. Run:
   ```
   python d:\Parkison\parkinsons_gui_no_database.py
   ```
   Or press `F5` in VS Code
4. GUI opens (1000x700 pixels, dark theme)

Usage
-----
1. Prediction Tab:
   - Manual Input:
     - Select "Manual Input"
     - Click "Load & Train Model"
     - Enter 22 vocal features (e.g., MDVP:Fo(Hz)=120.552)
     - Click "Test" for prediction, feature importance, and results
     - If Parkinson's detected, see "Visit 'Treatment' tab" prompt
   - Pre-trained Model:
     - Select "Pre-trained Model"
     - Click "Upload Data", select CSV (22 features, optional `status` column)
     - Click "Predict on Test" for batch results, metrics, visualizations
   - PDF Report:
     - After prediction, click "PDF Result"
     - Save as `d:\Parkison\report.tex`
     - Compile: `cd d:\Parkison && latexmk -pdf report.tex`
   - Export Results: Save predictions to GUI console
   - Help/About Us: View instructions and project details

2. Treatment Tab:
   - View expanded recommendation box (~760x520 pixels, ~97x67 pixel margins)
   - Scrollable advice on lifestyle, diet, medical management
   - Disclaimer: Consult a healthcare professional

3. Exit: Click "Exit", confirm to close

CSV Format
----------
- 22 features matching UCI dataset (e.g., MDVP:Fo(Hz),MDVP:Fhi(Hz),...)
- Optional `status` column (0=Healthy, 1=Parkinson's) for accuracy metrics
- Example:
  ```
  MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),...
  120.552,131.162,113.787,0.00607,...
  ```
- Save as `d:\Parkison\test_data.csv`

Troubleshooting
---------------
- PDF fails:
  - Update MikTeX (`MikTeX Console -> Updates`)
  - Check `report.log` for errors
  - Install missing packages via MikTeX Console
- Library errors:
  ```
  pip install <library>
  ```
- Tkinter missing: Reinstall Python
- CSV errors: Ensure 22 features, numerical values
- Internet: Required for UCI dataset
- Display scaling: Set to 100% if recommendation box size looks off
- Errors: Share traceback, Windows version (10/11), and steps (e.g., Manual Input, CSV upload)

License & Disclaimer
--------------------
- Open-source for educational purposes
- Medical Disclaimer: Not a substitute for professional medical advice. Consult a healthcare provider for diagnosis/treatment.

Contact
-------
For issues, share error details, screenshots, or any other problems, contact me on my email: Dhruvkumar2326@gmail.com


SCREENSHOTS:
![image](https://github.com/user-attachments/assets/e6630ea9-b5af-42b1-958e-17cb826b319f)
![image](https://github.com/user-attachments/assets/e2bb6bfe-f65b-4250-9944-94c46b7a8b57)
