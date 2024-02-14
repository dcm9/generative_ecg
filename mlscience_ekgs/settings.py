from pathlib import Path

# Data
data_path = Path(Path(__file__).resolve().parent, "Data")
raw_data_path = Path(data_path, "raw")
parsed_data_path = Path(data_path, "parsed")

# Raw data
ptb_path = Path(raw_data_path, "ptb")
ptb_xl_path = Path(raw_data_path, "ptb-xl")
mimic_note_path = Path(raw_data_path, "mimic-iv-note", "note")
mimic_wdb_path = Path(raw_data_path, "mimic-iv-wdb", "0.1.0", "waves")
mimic_ecg_path = Path(raw_data_path, "mimic-iv-ecg")

# Parsed data
mimic_ecg_parsed_path = Path(parsed_data_path, "mimic-iv-ecg")

# Result data
result_path = Path(data_path, "results")

code_path = Path(Path(__file__).resolve().parent, "Code")
exp_path = Path(code_path, "experiments")