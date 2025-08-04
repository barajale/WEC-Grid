"""
WEC-Grid database interface
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

# default location for the DB file
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB = os.path.join(_CURR_DIR, "WEC-GRID.db")

class WECGridDB:
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the WEC-Grid database handler.

        Args:
            db_path: Optional path to the SQLite database file.
                     If None, defaults to src/wecgrid/database/WEC-GRID.db
        """
        self.db_path = db_path or _DEFAULT_DB

    #TODO what does contextmanager do again? 
    @contextmanager
    def connection(self):
        """
        Context manager for SQLite connections. Commits on success,
        rolls back on exception, always closes.
        Usage:
            with db.connection() as conn:
                # use conn.cursor() ...
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            conn.close()
            
    # todo pull annd push wec and sim data, inital db with schema
    
    def initialize_db(self):
        pass
    
    
    