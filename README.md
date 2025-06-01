# IA-home-tasks

AI project focused on optimizing household tasks. It automates, organizes, and prioritizes domestic activities using intelligent algorithms to improve efficiency and save time.

---

## Project Description

This project leverages machine learning models and intelligent algorithms to help manage daily household tasks efficiently. By analyzing task difficulty, time, priority, weather conditions, and optimal hours, the system can schedule and prioritize chores, making home management smarter and less time-consuming.

---

## Dataset

The dataset includes household tasks with attributes such as:

- **task**: The name of the household activity (e.g., Barrer, Cocinar, Lavar platos, etc.)
- **difficult**: Numeric difficulty level (1-5)
- **time**: Estimated time in minutes to complete the task
- **priority**: Priority level (Baja, Media, Alta)
- **weather**: Preferred weather conditions (Soleado, Nublado, Lluvioso, Ventoso)
- **hour**: Suggested time of day (Mañana, Tarde, Noche)

This dataset is used to train models that predict optimal scheduling and task prioritization.

---

## Installation

Make sure you have Python 3.8+ installed. It's recommended to use a virtual environment.

```bash
# Create and activate virtual environment (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Or on Windows
python -m venv venv
venv\Scripts\activate
```

Install required packages with pip:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training the model

```bash
python main.py
```

Example command to organize tasks

``` bash
python predict_tasks.py "Cocinar,Lavar ropa,Organizar sala, Barrer, Cocinar, Lavar platos, Limpiar baño, Limpiar ventanas, Planchar ropa, Regar plantas, Sacar la basura, Tender cama"
```

---
