import re




numeric_pattern = re.compile(r'^[#\d]+$')
print(bool(re.search(r'\d', "sdas")))