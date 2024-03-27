class RoleMapper:
    def __init__(self, initial_mapping=None):
        self.role_mapping = initial_mapping if initial_mapping else {}

    def add_role(self, role):
        if role not in self.role_mapping:
            self.role_mapping[role] = role

    def map_role(self, role):
        if role in self.role_mapping:
            return self.role_mapping[role]
        else:
            self.add_role(role)
            return role

class DatasetAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset

    def analyze_roles(self):
        roles = set()
        for doc in self.dataset:
            roles.update(doc.get('roles', {}).keys())
        return roles

class RoleNormalizationPipeline:
    def __init__(self, dataset):
        self.dataset = dataset
        self.role_mapper = RoleMapper()

    def build_role_mapping(self):
        analyzer = DatasetAnalyzer(self.dataset)
        all_roles = analyzer.analyze_roles()
        for role in all_roles:
            self.role_mapper.add_role(role)

    def normalize_roles(self):
        for doc in self.dataset:
            roles = doc.get('roles', {})
            normalized_roles = {self.role_mapper.map_role(role): value for role, value in roles.items()}
            doc['roles'] = normalized_roles

def add_new_role_to_dataset(dataset, role):
    for doc in dataset:
        if 'roles' not in doc:
            doc['roles'] = {}
        doc['roles'][role] = None

def complex_functionality():
    # This function demonstrates additional complex functionality
    pass

# Sample dataset
dataset = [
    {"roles": {"winner": "team1", "victors": "team2"}},
    {"roles": {"champion": "team3", "victors": "team4"}}
]

# Additional roles to consider
additional_roles = ["defender", "runner-up", "participant"]

pipeline = RoleNormalizationPipeline(dataset)
pipeline.build_role_mapping()
pipeline.normalize_roles()

for role in additional_roles:
    add_new_role_to_dataset(dataset, role)

for doc in dataset:
    print(doc)
