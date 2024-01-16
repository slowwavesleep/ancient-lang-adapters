def process_list(input_list, num_elements=3):
    return input_list[:num_elements] + [""] * (num_elements - len(input_list))
