import sqlite3
import pandas as pd
from datetime import datetime


# ---- スキーマ定義 ----
TABLES = ["experiments", "metrics", "checkpoints"]
SCHEMA = {
    "experiments": 
        '''id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        model_type TEXT,
        config_json TEXT,
        created_at TEXT,
        state TEXT
        '''
    ,
    "metrics": 
        '''id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        timestamp TEXT,
        epoch INTEGER,
        train_loss REAL,
        val_loss REAL,
        train_accuracy REAL,
        val_accuracy REAL,
        learning_rate REAL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        '''
    ,
    "checkpoints": 
        '''id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        epoch INTEGER,
        accuracy REAL,
        path TEXT,
        timestamp TEXT,
        learning_rate REAL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        '''
    
}

class Database:
    def __init__(self, db_path):
        try:
            # データベース初期化
            self.conn = sqlite3.connect(db_path)
            cursor = self.conn.cursor()

            # テーブルが存在しない場合は作成
            for table_name in TABLES:
                # print(f"Creating table {table_name} if not exists...")
                # print(f"CREATE TABLE IF NOT EXISTS {table_name} ({SCHEMA[table_name]})")
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({SCHEMA[table_name]})")

            self.conn.commit()
            # self.conn.close()
            print(f"Database initialized at {db_path}")
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise e


    # ---- データベース操作 ----
    def insert_experiment(self, name, model_type, config_json, state="running"):
        cursor = self.conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO experiments (name, model_type, config_json, created_at, state) VALUES (?, ?, ?, ?, ?)",
            (name, model_type, config_json, created_at, state)
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_metrics(self, experiment_id, epoch, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate):
        cursor = self.conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO metrics (experiment_id, timestamp, epoch, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, timestamp, epoch, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate)
        )
        self.conn.commit()

    def insert_checkpoint(self, experiment_id, epoch, accuracy, path, learning_rate):
        cursor = self.conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO checkpoints (experiment_id, epoch, accuracy, path, timestamp, learning_rate) VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, epoch, accuracy, path, timestamp, learning_rate)
        )
        self.conn.commit()

    def update_state(self, experiment_id, state):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE experiments SET state = ? WHERE id = ?",
            (state, experiment_id)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
        print("Database connection closed.")

