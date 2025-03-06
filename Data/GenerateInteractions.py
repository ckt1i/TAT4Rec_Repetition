from openai import OpenAI
import random
import time
import json

client = OpenAI(api_key='', base_url="https://api.deepseek.com")

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


# 示例使用
user_interactions = [{'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B08P2DZB4X', 'rating': 5.0, 'timestamp': 1627391044559, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B086QY6T7N', 'rating': 5.0, 'timestamp': 1626614511145, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B08DHTJ25J', 'rating': 3.0, 'timestamp': 1626211245370, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B07RBSLNFR', 'rating': 5.0, 'timestamp': 1621184430697, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B07SLFWZKN', 'rating': 3.0, 'timestamp': 1619737501209, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B08JTNQFZY', 'rating': 5.0, 'timestamp': 1617904219785, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B08GLG6W8T', 'rating': 5.0, 'timestamp': 1613319236253, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B08M3C6LVS', 'rating': 3.0, 'timestamp': 1607339460872, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B07GHPCT6T', 'rating': 5.0, 'timestamp': 1598212476613, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B07KG1TWP5', 'rating': 5.0, 'timestamp': 1596473351088, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B07W397QG4', 'rating': 5.0, 'timestamp': 1593352422858, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B07GDQPG12', 'rating': 5.0, 'timestamp': 1547589843451, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B07J3GH1W1', 'rating': 5.0, 'timestamp': 1547589356557, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B01M7UMAUG', 'rating': 5.0, 'timestamp': 1508770624887, 'purchase': False}, {'user_id': 'AFSKPY37N3C43SOI5IEXEK5JSIYA', 'item_id': 'B00JMDPK8S', 'rating': 3.0, 'timestamp': 1404402463000, 'purchase': False}]

items = [{'title': 'Full Shine Clip in Hair Extensions 20 Inch Human Hair Clip in Extensions Balayage Color Gray Highlighted With Blonde Real Hair Extensions Clip in Human Hair 7 Pieces 100g', 'price': 0.0, 'average_rating': 4.6, 'rating_number': 2, 'features': '', 'description': '', 'store': 'Full Shine', 'categories': '', 'details': "{'Brand': 'Full Shine'; 'Material': 'Human Hair'; 'Extension Length': '20 Inches'; 'Hair Type': 'Straight'; 'Material Feature': 'Natural'; 'Is Discontinued By Manufacturer': 'No'; 'Product Dimensions': '7.48 x 4.72 x 1.38 inches; 6.98 Ounces'}", 'parent_asin': 'B08722DMPC'}, {'title': 'Townley Girl L.O.L Surprise! Mega Nail Set with Nail Dryer for Girls Kids Toddler |Perfect for Parties Sleepovers Makeovers| Birthday Gift for Girls above 3 Yrs', 'price': 0.0, 'average_rating': 4.4, 'rating_number': 25, 'features': '', 'description': '', 'store': 'Townley Girl', 'categories': '', 'details': '{}', 'parent_asin': 'B08CVS6BNQ'}, {'title': 'Ownest 200ml Matte Lip Gloss BaseLip Gloss Base Oil Material Lip Makeup Primers Non-Stick Lipstick Primer Lip Gloss Base for DIY Handmade Lip Balms Lip Gloss-200g', 'price': 0.0, 'average_rating': 3.4, 'rating_number': 71, 'features': '', 'description': '', 'store': 'Ownest', 'categories': '', 'details': "{'Brand': 'Ownest'; 'Item Form': 'Gel'; 'Package Information': 'Bottle'; 'Finish Type': 'Matte'; 'Specialty': 'Mattifying'; 'Package Dimensions': '7.99 x 5.2 x 1.18 inches; 10.37 Ounces'; 'UPC': '619191317851'; 'Manufacturer': 'Ownest'}", 'parent_asin': 'B08BF9PDC1'}]

timebefore = 1547589843451
timeafter = 1593352422858

if __name__ == '__main__':
    generator = GenerateInteractions()
    suggested_items = generator.generate_response(user_interactions, items , timeafter, timebefore)
    print(suggested_items)