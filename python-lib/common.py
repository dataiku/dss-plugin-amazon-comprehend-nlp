def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return new_name
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")