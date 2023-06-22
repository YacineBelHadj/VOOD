
from dynaconf import Dynaconf
from datetime import datetime
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['config/settings.toml', 'config/.secrets.toml','config/events.toml'],
)

def parse_datetime_strings(datetime_dict):
    parsed_dict = {}
    for key, value in datetime_dict.items():
        if isinstance(value, str):
            parsed_dict[key] = datetime.fromisoformat(value)
        else:
            parsed_dict[key] = value
    return parsed_dict

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
