'''

'''
class Deduplication:
    def __init__(self):
        pass

    def deduplicate(self, extracted_events):
        deduplicated_events = []
        seen_events = set()
        for event in extracted_events:
            event_hash = hash((event['time'], event['location'], frozenset(event['roles'].items())))
            if event_hash not in seen_events:
                seen_events.add(event_hash)
                deduplicated_events.append(event)
        return deduplicated_events


class ConflictResolution:
    def __init__(self):
        pass

    def resolve_conflicts(self, extracted_events):
        resolved_events = []
        for event in extracted_events:
            # Resolve conflicts in event time and location arguments
            time_counter = Counter()
            location_counter = Counter()
            for doc in event['documents']:
                time_counter.update([doc['time']])
                location_counter.update([doc['location']])
            most_common_time = time_counter.most_common(1)[0][0]
            most_common_location = location_counter.most_common(1)[0][0]

            # Resolve conflicts in event roles
            merged_roles = {}
            for doc in event['documents']:
                for role, entity in doc['roles'].items():
                    if role not in merged_roles:
                        merged_roles[role] = set()
                    merged_roles[role].add(entity)

            final_roles = {}
            for role, entities in merged_roles.items():
                final_roles[role] = sorted(entities)[0]  # Choosing the first entity for simplicity

            resolved_event = {
                'time': most_common_time,
                'location': most_common_location,
                'roles': final_roles
            }
            resolved_events.append(resolved_event)
        return resolved_events


if __name__ == "__main__":
    # Extracted events from multiple documents
    extracted_events = [
        {
            'time': '2023-01-01',
            'location': 'New York',
            'roles': {'subject': 'John', 'object': 'Mary'},
            'documents': [{'time': '2023-01-01', 'location': 'New York', 'roles': {'subject': 'John'}}]
        },
        {
            'time': '2023-01-01',
            'location': 'Los Angeles',
            'roles': {'subject': 'Alice', 'object': 'Bob'},
            'documents': [{'time': '2023-01-01', 'location': 'Los Angeles', 'roles': {'subject': 'Alice'}}]
        }
    ]

    deduplication_module = Deduplication()
    deduplicated_events = deduplication_module.deduplicate(extracted_events)

    conflict_resolution_module = ConflictResolution()
    resolved_events = conflict_resolution_module.resolve_conflicts(deduplicated_events)

    # Printing resolved events
    for event in resolved_events:
        print(event)
