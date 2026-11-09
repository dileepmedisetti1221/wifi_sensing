# Wi‑Fi Sensing & Occupancy Detection

A research‑grade prototype for **device‑free indoor occupancy classification** using RSSI measurements from existing Wi‑Fi access points on a college campus.

## Features
- **Live occupancy classification** – real‑time RSSI collection, feature extraction and Decision‑Tree inference displayed in a Streamlit dashboard.
- **Historical signal analysis** – per‑class (Empty / Moderate / Crowded) statistics, box‑plots and summary tables of the synthetic dataset.
- **Model evaluation** – confusion matrix, classification report, accuracy, precision, recall and F1‑score for the synthetic test set.
- **Downloadable experiment report** – a ZIP containing the confusion matrix image, test predictions CSV, class‑summary CSV and the original feature dataset.

## Project structure
```
wifi_sensing/
├─ .venv/                 # virtual environment (installed locally)
├─ data/
│   └─ features.csv       # synthetic feature records (Empty, Moderate, Crowded)
├─ models/
│   └─ occupancy_model.joblib   # trained DecisionTree model
├─ results/
│   ├─ confusion_matrix.png
│   └─ test_predictions.csv
├─ dashboard.py           # Streamlit app (live + analysis + evaluation + report)
├─ logger.py              # CLI for manual data collection
├─ train_model.py         # training & evaluation script
├─ README.md              # **this file**
└─ ...
```

## Getting started
### Prerequisites
- Windows 10/11 (the `netsh wlan show interfaces` command is Windows‑specific).
- Python 3.13 (or newer) – the project was built with Python 3.13.
- Git (for version control).

### Installation
```powershell
# Clone the repository (or use your local copy)
git clone https://github.com/dileepmedisetti1221/wifi_sensing.git
cd wifi_sensing

# Create a virtual environment and activate it
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or .\.venv\Scripts\activate.bat for cmd

# Install dependencies
pip install -r requirements.txt   # (or install manually)
# Required packages include: streamlit, pandas, numpy, scikit‑learn, joblib, plotly, streamlit‑autorefresh
```
> If a `requirements.txt` file does not exist, you can install the needed packages manually:
> ```powershell
> pip install streamlit pandas numpy scikit-learn joblib plotly streamlit-autorefresh
> ```

### Running the dashboard
```powershell
python -m streamlit run dashboard.py
```
Open the printed URL (usually `http://localhost:8501`) in a web browser.

- **Live tab** shows the current Wi‑Fi signal, computed statistics, and the model’s occupancy prediction.
- **Historical tab** visualises the synthetic dataset (mean RSSI per class, box‑plots, statistical tables).
- **Model Evaluation tab** displays the confusion matrix and a detailed classification report.
- **Download Report tab** creates a ZIP file (`wifi_occupancy_report.zip`) you can share.

## Data collection (optional)
The `logger.py` script can be used to record real‑world RSSI traces:
```powershell
python logger.py
```
You will be prompted for a room name and manual head‑count. The script writes a CSV file under `data/` for each recording session.

## Training / re‑training the model
If you add new recordings to `data/features.csv` (or generate new CSVs via `logger.py`), re‑train the model:
```powershell
python train_model.py
```
This will:
1. Load `data/features.csv`.
2. Split by session (one per class held‑out for testing).
3. Train a `DecisionTreeClassifier`.
4. Save the model to `models/occupancy_model.joblib`.
5. Produce `results/test_predictions.csv` and `results/confusion_matrix.png`.

## License
This project is provided for research and educational purposes. Feel free to adapt it, but **do not claim** that the synthetic model accuracy (≈ 98 %) represents real‑world performance.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your‑idea`).
3. Make your changes, commit, and push.
4. Open a Pull Request describing the enhancement.

---
*Prepared by the Antigravity AI assistant.*

