_registered_models = dict()


def build_model(config, model_name, **kwargs):
    if model_name not in _registered_models:
        raise ValueError(f'Unknown model: {model_name}')

    return _registered_models[model_name](config, **kwargs)


def register(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]
    _registered_models[model_name] = fn
    return fn
