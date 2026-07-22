from rdflib import Namespace, RDF
BRICK = Namespace("https://brickschema.org/schema/Brick#")
REF = Namespace("https://brickschema.org/schema/Brick/ref#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
QK = Namespace("http://qudt.org/vocab/quantitykind/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
S223 = Namespace("http://data.ashrae.org/standard223#")

def bind_prefixes(graph):
    """Associate common prefixes with the graph.

    :param graph: graph
    :type graph: rdflib.Graph
    """
    
    graph.bind("rdf", RDF)
    graph.bind("quantitykind", QK)
    graph.bind("qudt", QUDT)
    graph.bind("unit", UNIT)
    graph.bind("brick", BRICK)
    graph.bind("s223", S223)
    graph.bind('ref',REF)


namespace_dict = {
    "rdf": RDF,
    "quantitykind": QK,
    "qudt": QUDT,
    "unit": UNIT,
    "brick": BRICK,
    "s223": S223,
    "ref": REF,
}

prefix_dict = {value: key for key, value in namespace_dict.items()}

def get_prefixes(g):
    return "\n".join(
        f"PREFIX {prefix}: <{namespace}>"
        for prefix, namespace in g.namespace_manager.namespaces()
    )


def convert_to_prefixed(uri, g):
    try:
        prefix, uri_ref, local_name = g.compute_qname(uri)
        return f"{prefix}:{local_name}"
    except Exception as e:
        print(e)
        return uri
