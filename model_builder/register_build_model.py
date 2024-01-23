class Registry:
    _registry = dict()

    @classmethod
    def register_a_class(cls, cls_raw_obj):
        cls._registry[cls_raw_obj.__name__] = cls_raw_obj

    @classmethod
    def from_conf_create_model(cls, model_name: str, config: dict[str, any], **kwargs):
        if model_name not in cls._registry:
            raise ValueError(f'typo in model name or `{model_name}` is not implemented/registered yet')
        return cls._registry[model_name](config, **kwargs)
