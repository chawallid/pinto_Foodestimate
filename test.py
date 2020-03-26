import json
 
my_list = '{"left": "12", "centor": "12", "right": "12"}'
json_str = json.loads(my_list) 
query = [int(json_str["left"]),int(json_str["centor"]),int(json_str["right"])]

print(query)