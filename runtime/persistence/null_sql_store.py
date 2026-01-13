# runtime/persistence/null_sql_store.py

class NullSQLStore:
    def record_attempt(self, **kwargs):
        return "local-attempt"

    def record_artifact(self, *args, **kwargs):
        return None

    def append(self, *args, **kwargs):
        return None
