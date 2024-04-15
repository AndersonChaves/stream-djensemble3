from core.config import Config

def parse_dj_configurations(query: dict, config: dict, djconfig_key: str):
    djconfig = config["djensemble"][djconfig_key]
    if type(djconfig) is str:
        djconfig = Config(djconfig).data
    djconfig["config"] = djconfig_key
    djconfig["query"] = query
    parse_global_configurations(config, djconfig)
    return djconfig

def parse_global_configurations(config, djconfig: dict):
    # Include Global Configurations    
    for gconfig, value in config["global_configuration"].items():
        if gconfig == "min_purity_rate":
            djconfig["tiling"][gconfig] = value
        elif gconfig == "compacting_factor":
            djconfig["data_source"][gconfig] = value
        else:
            djconfig[gconfig] = value
