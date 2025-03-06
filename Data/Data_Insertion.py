from json import *
from GenerateInteractions import  *
import os
from datasets import load_dataset


Item_path_json = 'meta_All_Beauty.jsonl'

Raw_User_Data_path = 'Raw_User_Data.csv'

Raw_Item_Data_path = 'Raw_Item_Data.csv'

User_Data_path = 'User_Data.csv'

gap_days = 180

def read_interaction(line):
    data = loads(line)
    user_id = data['user_id']
    item_id = data['parent_asin']
    rating = data['rating']
    timestamp = int(data['timestamp'])
    purchase = bool(data['verified_purchase'])

    user_interaction = {
        'user_id': user_id,
        'item_id': item_id,
        'rating': rating,
        'timestamp': timestamp,
        'purchase': purchase
    }

    return user_interaction
        
def Read_user_csv_Data(User_path):
    with open(User_path, 'r') as f:
            users_interactions = []
            user_interactions = []
            for line in f:
                user_id, item_id, rating, timestamp, purchase = line.strip().split(',')
                user_interaction = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': float(rating),
                    'timestamp': int(timestamp),
                    'purchase': purchase == 'True'
                }
                if user_interactions and user_interactions[-1]['user_id'] != user_id:
                    users_interactions.append(user_interactions)
                    user_interactions = []
                user_interactions.append(user_interaction)
            if user_interactions:
                users_interactions.append(user_interactions)
        
    return users_interactions

def Read_user_json_Data():
    users_interactions = []
    usr_interactions = []
    user_id = 0
    data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
        
    for interaction in data:
        user_interaction = read_interaction(interaction)

        if user_interaction['user_id'] == user_id:
            usr_interactions.append(user_interaction)
        
        else:
            user_id = user_interaction['user_id']
            if len(usr_interactions) > 3:
                usr_interactions.sort(key=lambda x: x['timestamp'])
                users_interactions.append(usr_interactions)

            usr_interactions = []
            usr_interactions.append(user_interaction)

    return users_interactions

def Read_User_Data():
    if not os.path.exists(Raw_User_Data_path):
        users_interactions = Read_user_json_Data()
    else:
        users_interactions = Read_user_csv_Data(Raw_User_Data_path)
    return users_interactions

def read_item(line):
    data = loads(line)
    title = data.get('title', '')  
    price = data.get('price', 0.0)  
    average_rating = float(data.get('average_rating', 0.0))  
    rating_number = int(data.get('rating_number', 0))  
    features = ','.join(data.get('features', []))  
    description = ','.join(data.get('description', []))  
    store = data.get('store', '') 
    categories = ','.join(data.get('categories', []))  
    details = str(data.get('details', '')) 
    item_id = data.get('parent_asin', '') 

    item = {
        'title': title,
        'price': price,
        'average_rating': average_rating,
        'rating_number': rating_number,
        'features': features,
        'description': description,
        'store': store,
        'categories': categories,
        'details': details,
        'parent_asin': item_id
    }
    return item

def Read_item_csv_Data(Data_path):
    items = []
    with open(Data_path, 'r') as f:
        for line in f:
            title, price, avg_rating, rating_num, features, desc, store, cats, details, parent_asin = line.strip().split(',', 9)
            item = {
                'title': title,
                'price': price if price not in ('', 'None') else 0.0,
                'average_rating': float(avg_rating),
                'rating_number': int(rating_num),
                'features': features,
                'description': desc,
                'store': store,
                'categories': cats,
                'details': details,
                'parent_asin': parent_asin
            }
            items.append(item)
    return items

def Read_item_json_Data(Data_path):
    with open(Data_path, 'r') as f:
        data = f.readlines()
        items = []
        for line in data:
            item = read_item(line)
            items.append(item)
    return items

def Read_Item_Data():
    if not os.path.exists(Raw_Item_Data_path):
        data = Read_item_json_Data(Item_path_json)
    else:
        data = Read_item_csv_Data(Raw_Item_Data_path)
    return data


def Save_User_CSV(users_interactions, User_Data_path):
    with open(User_Data_path, 'w') as f:
        for user_interactions in users_interactions:
            for interaction in user_interactions:
                if not isinstance(interaction, dict):
                    print(f"Skipping invalid interaction: {interaction}, type: {type(interaction)}")
                    continue
                f.write(f"{interaction['user_id']},{interaction['item_id']},{interaction['rating']},{interaction['timestamp']},{interaction['purchase']}\n")

def Save_Item_CSV(items , path):
    with open(path, 'w') as f:
        for item in items:
            try:
                title = str(item.get('title', '')).replace(',', '').replace('\n', ' ')
                price = str(item.get('price', 0.0))
                average_rating = str(item.get('average_rating', 0.0))
                rating_number = str(item.get('rating_number', 0))
                features = str(item.get('features', '')).replace(',', ';')
                description = str(item.get('description', '')).replace(',', ';')
                store = str(item.get('store', '')).replace(',', '')
                categories = str(item.get('categories', '')).replace(',', ';')
                details = str(item.get('details', '')).replace(',', ';')
                parent_asin = str(item.get('parent_asin', ''))

                f.write(f"{title},{price},{average_rating},{rating_number},{features},"
                        f"{description},{store},{categories},{details},{parent_asin}\n")
            except Exception as e:
                print(e)


def Count_Timegap(days, timeafter, timebefore):
    # count time gaps between user interactions
    if timeafter - timebefore > days*86400*1000:
        return True
    else:
        return False

def Data_Insertion(users_interactions,items):

    generator = GenerateInteractions()

    for user_interactions in users_interactions:

        timebefore = user_interactions[0]['timestamp']

        for interaction in user_interactions: 

            if not isinstance(interaction, dict):

                print(f"Skipping invalid interaction: {interaction}, type: {type(interaction)}")
                
                continue

            tmp_file = open('tmp_data.csv', 'a')
            
            tmp_file.write(f"{interaction['user_id']},{interaction['item_id']},{interaction['rating']},{interaction['timestamp']},{interaction['purchase']}\n")

            if len(user_interactions) < 5:
                continue

            # check the time gaps between each user's interactions
            timeafter = interaction['timestamp']

            if 'pesudo' in interaction['item_id']:  # 检查是否有伪造交互
                try:
                    user_interactions.sort(key=lambda x: x['timestamp'])
                except Exception as e:
                    print(f"Sort error: {e}, user_interactions: {user_interactions}")
                finally:
                    print("User interactions are Solved")
                    break
                
            if Count_Timegap(gap_days, timeafter, timebefore):
                try :
                    
                    pesudo_interactions = generator.pesudo_generation(user_interactions, items, timeafter, timebefore)
                
                except Exception as e:

                    print(f"Error: {e}, user_interactions: {user_interactions}, timeafter: {timeafter}, timebefore: {timebefore}")
                    
                    pesudo_interaction = None

                if pesudo_interactions:

                    for pesudo_interaction in pesudo_interactions:

                        tmp_file.write(f"{pesudo_interaction['user_id']},{pesudo_interaction['item_id']},{pesudo_interaction['rating']},{pesudo_interaction['timestamp']},{pesudo_interaction['purchase']}\n")
                    
                    user_interactions.extend(pesudo_interactions)
                    
                    if not isinstance(interaction, dict):
                        
                        print(f"Skipping invalid interaction: {interaction}, type: {type(interaction)}")

            tmp_file.close()
            
            timebefore = timeafter

    return users_interactions


def main():
    
    Raw_user_interactions = Read_User_Data()

    if not os.path.exists(Raw_User_Data_path):
        Save_User_CSV(Raw_user_interactions, Raw_User_Data_path)

    items = Read_Item_Data()
    
    if not os.path.exists(Raw_Item_Data_path):
        Save_Item_CSV(items, Raw_Item_Data_path)

#    users_interactions = Data_Insertion(Raw_user_interactions, items)

#    Save_User_CSV(users_interactions , User_Data_path)

if __name__ == '__main__':
    main()