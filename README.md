# 🏘️ AI Village Layout Planner with Q-Learning

This project is an intelligent system that automatically generates and evaluates housing layouts on a grid using Q-Learning. It optimizes the placement of roads, houses, and green spaces based on profitability, accessibility, and urban planning constraints.

## 📁 Project Structure

```
├── jecsu34.py              # Main AI engine using Q-learning (multi-grid training, SQLite Q-table)
├── jecsun.py               # Simplified version for layout optimization and Streamlit use
├── makecsvSQmaps.py        # CSV map generator with rotation/flip variations
├── reward_calculator.py    # Core logic for scoring the layout (with bonuses and penalties)
├── memory_utils.py         # RAM usage checks and cleaning utilities
├── error_handling.py       # Decorators and custom exceptions for safe execution
├── backup_utils.py         # Auto-backup and recovery system for Q-table and database
├── config.py               # Central config file (AI settings, file paths, thresholds)
├── config_logging.py       # Logging configuration
├── village-ai-dashboard.py # Streamlit web UI for AI training, visualization, and profit analysis
```

## 🚀 Features

- **Q-Learning AI** with adaptive exploration and learning rate
- **Parallel grid training** using multiprocessing
- **Reward function** with bonuses (e.g. edge housing, street clusters) and penalties (e.g. unconnected houses)
- **Auto-backup system** for Q-table and SQLite DB
- **Interactive dashboard** built with Streamlit
- **Custom CSV map ingestion** via CLI input and image transformation
- **Memory usage control** for resource-limited environments

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-village-planner.git
cd ai-village-planner
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> 🧩 Includes: `numpy`, `pandas`, `psutil`, `flask`, `streamlit`, `tabulate`, `filelock`, `scipy`, `requests`

## 💡 How to Use

### ✅ Training the AI

```bash
python jecsu34.py
```

This will:
- Load or create a grid
- Train AI on multiple E positions
- Auto-save results to `q_table.db` and `q_table.json`

### ✅ Launch Web Dashboard

```bash
streamlit run village-ai-dashboard.py
```

You'll be able to:
- Choose grid size and E location
- Visualize AI-generated layouts
- Analyze economic profitability

### ✅ Create Custom Maps

```bash
python makecsvSQmaps.py
```

- Manually input a grid layout
- Script generates all rotated/flipped versions
- Saves non-duplicate CSVs for training

## 🧠 AI Concepts Used

- **Q-Learning** with decaying epsilon and alpha
- **Grid state encoding** using localized 3x3 patterns
- **Reward Shaping**:
  - ✅ Bonuses: connected roads, edge houses, housing clusters
  - ❌ Penalties: unconnected buildings, missing roads

## 📦 Output Files

- `q_table.json` – Serialized Q-Table (for inference or dashboard)
- `q_table.db` – SQLite version of the Q-table
- `/logs/` – AI process logs
- `/data/maps/CSV/` – Training map sources
- `/data/backups/` – Backup Q-tables (auto-pruned after 7 days)

## 🛡️ Reliability & Fault Tolerance

- **Error Handling**: Decorators wrap risky functions with logs and recoveries
- **Memory Check**: Training is aborted if system memory is insufficient
- **Locking System**: Prevents concurrent writes to Q-table
- **Backup**: Every session auto-saves timestamped backups

## 👨‍💻 Authors & Maintainers

- Project by: Purapan Poolphon
- Intern at: Supalai Public Company Limited
- Supervisor: Dr. Panudech Chumyen

## 📄 License

This project is for educational use. For commercial licensing, please contact the author.
