
from dynaconf import Dynaconf
from pathlib import Path
import json
from datetime import datetime , timedelta
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['config/settings.toml',
                    'config/.secrets.toml',
                    'config/events.toml',
                    'config/model_params.toml',
                    'config/preprocessing.toml',]
)

def parse_datetime_strings(datetime_dict):
    parsed_dict = {}
    for key, value in datetime_dict.items():
        if isinstance(value, str):
            parsed_dict[key] = datetime.fromisoformat(value)
        else:
            parsed_dict[key] = value
    return parsed_dict



def parse_preprocessing(data : dict):
    data['frame_size'] = timedelta(data['frame_size'])
    data['frame_step'] = timedelta(data['frame_step'])
    return data
def load_metadata(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)  

parser = {'datetime': parse_datetime_strings,
          'preprocessing': parse_preprocessing}

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
