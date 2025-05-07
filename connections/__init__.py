import connections.mysql_connections as mysql_connections
from functools import lru_cache

@lru_cache()
def getConnections():
    # This function is a placeholder for the actual implementation that retrieves database connections.
    # You can replace the return statement with your own logic to fetch and return the connections.
    return mysql_connections.engine_raw_data, mysql_connections.engine_clean_data



connectionsdb = getConnections()