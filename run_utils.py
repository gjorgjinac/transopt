def get_argument_elements_from_list(argument_list_str, cast_to_int=False):
    if cast_to_int:
        return [int(a) for a in argument_list_str.split('-')] if '-' in argument_list_str else [int(argument_list_str)]
    else:
        return argument_list_str.split('-') if '-' in argument_list_str else [argument_list_str]