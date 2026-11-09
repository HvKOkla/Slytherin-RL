from . import obs_v1
from . import obs_v1_1 

OBS_REGISTRY = {
    "v1": obs_v1.get_observation,
    "v1_1": obs_v1_1.get_observation, 
}

def get_obs_function(version="v1"):
    if version not in OBS_REGISTRY:
        raise ValueError(f"Version d'observation inconnue : {version}")
    return OBS_REGISTRY[version]