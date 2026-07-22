Potential functions

sparql_query - Execute raw SPARQL queries against the building's knowledge graph. This gives maximum flexibility for complex queries and allows the LLM to construct sophisticated queries when needed. Return first 10 results.

describe_entity - Retrieve all properties and relationships for a specific entity (equipment, space, point, etc.) by URI or label. This helps the LLM understand individual components deeply.

get_building_summary - Classes and their counts. Relationships and their counts.

find_entities_by_type - List all entities of a given Brick class (e.g., all VAVs, all temperature sensors). Essential for understanding what exists in the building.



Maybe
get_hierarchy - Traverse spatial or equipment hierarchies (e.g., "show me everything in HVAC Zone 3" or "what's the structure under AHU-1").

get_relationships - Query specific relationship types (feeds, hasPoint, hasLocation, etc.) either globally or filtered by entity. Critical for understanding system topology.

get_similar_entities - fuzzy search for entities/classes


Test the effects of various tools on token costs and levels of success. 