from datasets import load_dataset
import os
from openai import OpenAI
import random
import time
import json

client = OpenAI(api_key='sk-8d362c7dcc8b4409af68c405d6080286', base_url="https://api.deepseek.com")

class GenerateInteractions:
    def __init__(self):
        self.client = client

    def generate_response(self, user_interactions, items , timeafter, timebefore):
        # 模拟流行信息
        popular_info = "Currently, items in the beauty and personal care category are trending. Users preferences will influenced by the trends."
        
        # 解析用户交互序列
        user_interactions_parsed = (
            "The user interaction sequence contains the following fields for each interaction:\n"
            "- user_id: Unique identifier for the user.\n"
            "- item_id: Unique identifier for the item interacted with.\n"
            "- rating: The rating given by the user (ranging from 1.0 to 5.0).\n"
            "- timestamp: The time of the interaction in milliseconds since epoch.\n"
            "- purchase: A boolean indicating whether the interaction resulted in a purchase.\n"
            f"Here is the user's interaction history:\n{user_interactions}"
        )

        # 解析物品列表
        items_parsed = (
            "The available items contain the following fields for each item:\n"
            "- title: The title or name of the item.\n"
            "- price: The price of the item.\n"
            "- average_rating: The average rating of the item.\n"
            "- rating_number: The number of ratings the item has received.\n"
            "- features: Additional features of the item.\n"
            "- description: A description of the item.\n"
            "- store: The store or brand offering the item.\n"
            "- categories: The categories the item belongs to.\n"
            "- details: Additional details about the item.\n"
            "- parent_asin: A unique identifier for the item.\n"
            f"Here are the available items:\n{items}"
        )

        # 生成系统提示
        system_prompt = (f"{popular_info} Based on the user's interaction history and the popular trends in the time gap from {timebefore} to {timeafter}. The time is recorded in unix time counted by ms.\n"    
                        f"Consider the informations above, If there's possible items, create one or two different pesudo interactions during the time gap in json form."
                        "The JSON should be an object with an 'interactions' key containing a list of interaction objects with fields: 'user_id', 'item_id', 'rating', 'timestamp', 'purchase'.\n"
                        "If no suitable item is found, return {'interactions': []}.\n"
                        "Example response: {'interactions': [{'user_id': 'user123', 'item_id': 'item456', 'rating': 4.0, 'timestamp': 1600000000000, 'purchase': false}]}"
        )
        # 生成用户提示
        user_prompt = f"User interactions: {user_interactions_parsed}.  Selected items: {items_parsed}."
        try: 
            # 调用Deepseek API
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                response_format={
                    'type': 'json_object'
                }
            )

            # 解析API响应
            suggested_items = response.choices[0].message.content

            # 检查是否为空或无效
            if not suggested_items or suggested_items.strip() == "":
                print("API returned empty response")
                return None
            
            # 解析 JSON 字符串为字典
            try:
                parsed_response = json.loads(suggested_items)
                # 检查是否包含 'interactions' 键
                if isinstance(parsed_response, dict) and "interactions" in parsed_response:
                    # 如果是空交互列表，返回 None（表示无建议）
                    if not parsed_response["interactions"]:
                        return None
                    return parsed_response  # 返回解析后的字典
                else:
                    print("Response does not match expected format")
                    return None
            except json.JSONDecodeError as e:
                print(f"Failed to parse response as JSON: {e}")
                print(f"Raw response: {suggested_items}")
                return None
            

        except Exception as e:
            print(f"API request failed: {e}")
            return None

    @staticmethod
    def random_selection(items):

        selected_items = []

        for i in range(10):

            random.seed(time.time())

            selected_items.append(random.choice(items))

        return selected_items   


    def pesudo_generation(self, user_interactions , items , timeafter, timebefore):
        # Select random items to insert into the user interactions
        selected_items = self.random_selection(items)

        # Generate Response for the selected items
        for i in range(1):
            response = self.generate_response(user_interactions, selected_items, timeafter, timebefore)
            if response != None:
                break
        
        if response == None:
            return []
        
        print("Generated Pesudo interactions")
    
        # 检查 interactions 是否包含 'interactions' 键
        try:
            new_interactions = []
            for suggestion in response["interactions"]:
                if suggestion['timestamp'] > min(timebefore , timeafter) and suggestion['timestamp'] < max(timebefore , timeafter):
                    suggestion['item_id'] = suggestion['item_id'] + "_pesudo"
                    new_interactions.append(suggestion)
            return new_interactions
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse suggestions: {e}")
            return []
        except KeyError as e:
            print(f"Invalid response structure: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []



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

def read_interaction(data):

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


def Count_Timegap(days, timeafter, timebefore):
    # count time gaps between user interactions
    if timeafter - timebefore > days*86400*1000:
        return True
    else:
        return False


class AmazonProcessing():
   
    def __init__(self , user_raw_path , item_data_path , user_data_path ,  data_path , user_type , item_type , min_interactions=5 , gap_days=30):
        
        self.userdata = []
        self.itemdata = []
        
        self.user_raw_path = user_raw_path
        self.item_data_path = item_data_path
        self.user_data_path = user_data_path

        self.tmp_path = 'Data/tmp.csv'

        self.data_path = data_path
        self.user_type = user_type
        self.item_type = item_type

        self.min_interactions = min_interactions
        self.gap_days = gap_days

        self.raw_interactions = []
        self.items = []
        self.interactions = []


    def extract_user_interactions(self):
        
        users_interactions = []
        user_interactions = []

        user_id = 0

        for data in self.userdata:

            user_interaction = read_interaction(data)

            if user_interaction['user_id'] == user_id:
                user_interactions.append(user_interaction)
            
            else:
                user_id = user_interaction['user_id']
                
                if len(user_interactions) > self.min_interactions:
                    user_interactions.sort(key=lambda x: x['timestamp'])
                    users_interactions.append(user_interactions)

                user_interactions = []

        return users_interactions

    def extract_item_data(self):
        
        data = self.itemdata
        items = []

        for line in data:

            title = line.get('title', '')  
            price = line.get('price', 0.0)  
            average_rating = float(line.get('average_rating', 0.0))  
            rating_number = int(line.get('rating_number', 0))  
            features = ','.join(line.get('features', []))  
            description = ','.join(line.get('description', []))  
            store = line.get('store', '') 
            categories = ','.join(line.get('categories', []))  
            details = str(line.get('details', '')) 
            item_id = line.get('parent_asin', '') 

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

            items.append(item)

        return items


    def data_extraction(self):
        # Extracting user interactions and item data from dataset
        if not os.path.exists(self.user_raw_path):

            print("Raw user's csv not foun, load from dataset")
            self.userdata = load_dataset(self.data_path, self.user_type, split="full", trust_remote_code=True)
            self.raw_interactions = self.extract_user_interactions()

            print("Saving raw user data to csv files")
            Save_User_CSV(self.raw_interactions, self.user_raw_path)

        else: 

            self.raw_interactions = Read_user_csv_Data(self.user_raw_path)

        if not os.path.exists(self.item_data_path):
            
            print("Item's csv not found, load from dataset")
            self.itemdata = load_dataset(self.data_path, self.item_type, split="full", trust_remote_code=True)
            self.items = self.extract_item_data()

            print("Saving item data to csv files")
            Save_Item_CSV(self.items, self.item_data_path)

        else:
            
            self.items = Read_item_csv_Data(self.item_data_path)

        return self.raw_interactions, self.items


    def data_insertion(self): 
        
        if self.raw_interactions == [] or self.items == []:
            
            users_interactions , items =  self.data_extraction()

        else:

            users_interactions = self.raw_interactions
            items = self.items

        # Generate Pesudo interactions
        generator = GenerateInteractions()

        for user_interactions in users_interactions:

            timebefore = user_interactions[0]['timestamp']

            for interaction in user_interactions: 

                if not isinstance(interaction, dict):

                    print(f"Skipping invalid interaction: {interaction}, type: {type(interaction)}")
                    
                    continue

                tmp_file = open(self.tmp_path, 'a')
                
                tmp_file.write(f"{interaction['user_id']},{interaction['item_id']},{interaction['rating']},{interaction['timestamp']},{interaction['purchase']}\n")

                if len(user_interactions) < self.min_interactions:
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
                    
                if Count_Timegap(self.gap_days, timeafter, timebefore):
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

        
    def data_processing(self):
        # Process the data
        if not os.path.exists(self.user_raw_path) or not os.path.exists(self.item_data_path):
            print("csv files not found, extracting data from dataset")
            self.data_extraction()

        else:
            print("Reading data from csv files")
            self.raw_interactions = Read_user_csv_Data(self.user_raw_path)
            self.items = Read_item_csv_Data(self.item_data_path)
        
        print("Inserting Pesudo interactions")
        users_interactions = self.data_insertion()

        print("Saving user augmented data to csv files")
        Save_User_CSV(users_interactions , self.user_data_path)



# Samples

def main():
    Raw_path = 'Data/Raw_User_Data.csv'
    Item_path =  'Data/Item_Data.csv'
    User_path =  'Data/User_Data.csv'

    Amazon_Data = AmazonProcessing(Raw_path , Item_path , User_path , 'McAuley-Lab/Amazon-Reviews-2023', 'raw_review_All_Beauty', 'raw_meta_All_Beauty', 0, 30)

    Amazon_Data.data_processing()

if __name__ == "__main__":
    main()