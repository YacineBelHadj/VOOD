
from dynaconf import Dynaconf
from config.events import events

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['config/settings.toml', 'config/.secrets.toml','config/events.py'],
)
events =events

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
