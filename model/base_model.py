from model_builder import Registry


class BaseModel:
    def __init__(self, config: dict[str, any]):
        self.heads = config['heads']

    def __init_subclass__(cls, **kwargs):
        super.__init_subclass__(**kwargs)
        Registry.register_a_class(cls)

    def __repr__(self):
        return f''