def get_key(json, key):
    """
    Method to parse json of keys extracted, returns list of values
    Args:
        json: the full json object
        key: the key to search by, returns values for this key
    """
    return list(findkeys(json, key))

def findkeys(node, kv):
    """
    Method to get json from nested json starting from root
    Args:
        node: the json root to start at
        kv: the key to search
    """
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x