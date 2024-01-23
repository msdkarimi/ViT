_registered_models = dict()


def register(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]
    _registered_models[model_name] = fn
    return fn
