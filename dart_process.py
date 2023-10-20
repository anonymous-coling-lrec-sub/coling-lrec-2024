import re

def get_nodes(n):
    n = n.strip()
    n = n.replace('(', '')
    n = n.replace('\"', '')
    n = n.replace(')', '')
    n = n.replace(',', ' ')
    n = n.replace('_', ' ')

    #n = ' '.join(re.split('(\W)', n))
    # n = unidecode.unidecode(n)
    # n = n.lower()

    return n

def get_relation(n):
    n = n.replace('(', '')
    n = n.replace(')', '')
    n = n.strip()
    n = n.split()
    n = "_".join(n)
    n = n.lower()
    return n

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            #new_d.append(t.lower())
            new_d.append(t)
    return new_d

def process_triples(mtriples):
    nodes = []

    for m in mtriples:
        ms = m
        n1 = ms[0]
        n2 = ms[2]
        nodes1 = get_nodes(n1)
        nodes2 = get_nodes(n2)

        edge = get_relation(ms[1])

        edge_split = camel_case_split(edge)
        edges = ' '.join(edge_split)

        nodes.append('<H>')
        nodes.extend(nodes1.split())

        nodes.append('<R>')
        nodes.extend(edges.split())

        nodes.append('<T>')
        nodes.extend(nodes2.split())

    return " ".join(nodes)