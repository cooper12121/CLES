

import json
import os
from collections import Counter
from collections import defaultdict

def read_jsonline(file_path):
    result = []
    with open(file_path,'r',encoding='utf-8')as f:
        for line in f:
            result.append(json.loads(line))
    return result

def count_entity_role_result(dict:dict,dict_result:dict):
    new_result = defaultdict(int)
    for key,values in dict_result.items():
        new_result[key]={
              key: dict[key], #计算key自身的出现次数
         }
        for value in values:
            new_result[key][value] = dict[value]
        new_result[key]['max_key'] = max(new_result[key],key=new_result[key].get)
    return new_result


def get_collections(data_list:dict)-> dict:
    grouped_data = defaultdict(list)
    for item in data_list:
        coref_chain = item['coref_chain']
        grouped_data[coref_chain].append(item)
    return grouped_data

def normalize_entity(data_list:list,entity_dict_count:dict,role_dict_count:dict):
  
    collections = get_collections(data_list)
    collection_event = []
    for event_id, events_list in collections.items():
        entity_set = set()
        for event_doc in events_list:
            try:
                events = json.loads(event_doc['new_events'].strip().replace('\'', '\"'))
                for event in events:
                    arguments = []
                    for mention in event['arguments']:
                        entity  = mention.get('mention',None) if mention.get('mention',None) else mention.get('argument',None)

                        entity_unqiue, role_unique = None, None
                        if entity_dict_count.get(entity,None):
                            entity_unqiue = entity_dict_count[entity]['max_key']
                            mention['mention'] = entity_unqiue

                        role  = mention.get('role',None)
                        if role_dict_count.get(role,None):
                            role_unique = role_dict_count[role]['max_key']
                            mention['role'] = role
                        
                        if entity_unqiue and entity_unqiue not in entity_set:
                                entity_set.add(entity_unqiue)
                                arguments.append(mention)
                    event['arguments'] = arguments


                    collection_event.append(event)
            except Exception as e:
                print(e,event_doc['coref_chain'])
                continue
   




def merge_events(*event_lists,role_dict_count:dict):
    all_events = []
    for events in event_lists:
        all_events.extend(events)
    
    merged_events = {}
    for event in all_events:
        key = (event['type'], event['trigger'])
        if key not in merged_events:
            merged_events[key] = {
                'type': event['type'],
                'trigger': event['trigger'],
                'arguments': []
            }
        merged_events[key]['arguments'].extend(event['arguments'])
    
    for key, event in merged_events.items():
        mention_to_role = {}
        for arg in event['arguments']:
            mention = arg['mention']
            role = arg['role']
            if mention in mention_to_role:
                
                existing_role = mention_to_role[mention]
                if role_dict_count[role].get("max_key", 0) > role_dict_count[existing_role].get("max_key", 0):
                    mention_to_role[mention] = role
            else:
                mention_to_role[mention] = role
        
        event['arguments'] = [{'mention': m, 'role': r} for m, r in mention_to_role.items()]
    return list(merged_events.values())


if __name__ == "__main__":
    
    file_dir = "./CLES/raw"
    for file in ['train_process.json','dev_process.json','test_process.json']:
        if ".jsonl" in file:
                data_list = read_jsonline(os.path.join(file_dir,file))
        elif ".json" in file:
                data_list = json.load(open(os.path.join(file_dir,file)))
        entity_dict_count = json.load(open("./CLES/count/entity_dict_count.json"))
        role_dict_count = json.load(open("./CLES/count/role_dict_count.json"))
        
        collections = get_collections(data_list)
       


        output_name = file.split('_')[0] + "_collection.json"

        json.dump(collections,open(os.path.join("./CLES/collections",output_name),'w'),ensure_ascii=False)

   
