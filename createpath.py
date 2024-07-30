import datetime
import os
from pathlib import Path
log_dir=Path("logs").mkdir(parents=True, exist_ok=True)
log_file="log_dir"+"log_file"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")