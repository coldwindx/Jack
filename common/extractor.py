
### extract features from every layers in network
import collections

class FeatureExtractor:
    def __init__(self):
        self.features = collections.defaultdict(list)

    def create_hook(self, name):
        def hook(module, input, output):
            self.features[name].append(output)
        return hook
    
    def register(self, model, names):
        for (name, module) in model.named_modules():
            if name in names:
                hook = self.create_hook(name)
                module.register_forward_hook(hook)

    def get_features(self, name):
        return self.features[name]