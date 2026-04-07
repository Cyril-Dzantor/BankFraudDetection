from neo4j import GraphDatabase
from app.core.config import settings

class GraphManager:
    def __init__(self):
        self.driver = None

    def connect(self):
        if not self.driver:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                connection_timeout=5.0
            )

    def get_session(self):
        self.connect()
        return self.driver.session(database=settings.NEO4J_DATABASE)

    async def sync_transaction(self, alert_data: dict):
        """
        Syncs a transaction and its entities (Account, Device, IP) to Neo4j.
        """
        self.connect()
        # Extract entities from the alert data
        # Note: We'll need to expand our data model slightly to include real IPs/Devices
        # for more robust link analysis in production.
        
        query = """
        MERGE (a:Account {id: $account_id})
        MERGE (d:Device {id: $device_id})
        MERGE (l:Location {name: $location})
        
        MERGE (a)-[:USED_DEVICE]->(d)
        MERGE (a)-[:TRANSACTED_FROM]->(l)
        
        CREATE (t:Transaction {
            id: $txn_id,
            amount: $amount,
            timestamp: datetime(),
            riskLevel: $risk_level,
            score: $score
        })
        
        MERGE (a)-[:PERFORMED]->(t)
        MERGE (t)-[:ON_DEVICE]->(d)
        MERGE (t)-[:IN_LOCATION]->(l)
        """
        
        params = {
            "account_id": alert_data.get("acctStart"),
            "device_id": alert_data.get("device"),
            "location": alert_data.get("location"),
            "txn_id": alert_data.get("id"),
            "amount": alert_data.get("amount"),
            "risk_level": alert_data.get("riskLevel"),
            "score": alert_data.get("score")
        }
        
        with self.get_session() as session:
            session.run(query, params)

graph_manager = GraphManager()
