from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "sensor_bowtie"))
with driver.session() as session:
    result = session.run("RETURN 'Neo4j is working!' AS msg")
    print(result.single()["msg"])
