

# based on:
# https://stackoverflow.com/questions/38491318/python-replace-keys-in-a-nested-dictionary
def replace_values(data_dict, value_dict):
    new_dict = {}
    if isinstance(data_dict, list):
        dict_value_list = list()
        for inner_dict in data_dict:
            dict_value_list.append(replace_values(inner_dict, value_dict))
        return dict_value_list
    else:
        for key in data_dict.keys():
            value = data_dict[key]
            new_value = value_dict.get(key, value)
            if isinstance(value, dict) or isinstance(value, list):
                new_dict[key] = replace_values(value, value_dict)
            else:
                new_dict[key] = new_value
        return new_dict
    return new_dict
