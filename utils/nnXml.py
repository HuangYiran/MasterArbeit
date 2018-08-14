try:
    import xml.etree.cElementTree as et
except:
    import xml.etree.ElementTree as et


def read_exp_list(doc):
    """
    read all the params under the <e> tag
    input: the file that store the parameters of the experiment
    output: parameter dict
    """
    params_list = []
    tree = et.ElementTree(file = doc)
    for e in tree.iter(tag = 'e'):
        params = {}
        for item in e:
            value = item.text.strip()
            if 'type' not in item.attrib:
                print "type attrib is not set, set the value to string type"
            if item.attrib['type'] == "int":
                if value == "0":
                    value = None
                else:
                    value = int(value)
            elif item.attrib['type'] == "float":
                value = float(value)
            elif item.attrib['type'] == "string":
                value = value
            elif item.attrib['type'] == "boolean":
                if value == "True":
                    value = True
                else:
                    value = False
            elif item.attrib['type'] == "none":
                value = None
            else:
                print "unrecognize type, set the value to None"
                value = None
            params[item.tag] = value
        params_list.append(params)
    return params_list
