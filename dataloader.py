"""
Database module for loading survey data into Polars DataFrames.
"""
import polars as pl
from typing import Optional
from connect_db import get_connection

class SurveyDatabase:
    """Handle database connections and data loading."""
    
    def __init__(self, conn: str):
        """
        Initialize database connection.
        """
        self.conn = get_connection()
    
    def load_introductory(self) -> pl.DataFrame:
        """
        Load introductory data (once per survey).
        
        Returns:
            Polars DataFrame with introductory data
        """
        query = """
        SELECT 
            id
        FROM introductory
        """
        return pl.read_database(query, self.conn)
    
    def load_home_training(self) -> pl.DataFrame:
        """
        Load home training data (yearly per child).
        
        Returns:
            Polars DataFrame with home training data
        """
        query = """
        SELECT 
            id,
            introductory_id,
            training_methods_therapies,
            devices,
            other_training_methods_therapies
        FROM home_training
        """
        return pl.read_database(query, self.conn)
    
    def load_intensive_therapies(self) -> pl.DataFrame:
        """
        Load intensive therapies data (yearly per child).
        
        Returns:
            Polars DataFrame with intensive therapies data
        """
        query = """
        SELECT 
            id,
            introductory_id,
            participate_therapies_neurohabilitation,
            neurohabilitation_centers,
            methods_applied_during_intense_training,
            medical_treatments
        FROM intensive_therapies
        """
        return pl.read_database(query, self.conn)
    
    def load_motorical_development(self) -> pl.DataFrame:
        """
        Load motorical development data (yearly per child).
        
        Returns:
            Polars DataFrame with motorical development data
        """
        query = """
        SELECT 
            id,
            introductory_id,
            gross_motor_development,
            fine_motor_development,
            motorical_impairments_lower,
            motorical_impairments_upper
        FROM motorical_development
        """
        return pl.read_database(query, self.conn)
    
    def load_all_data(self) -> dict[str, pl.DataFrame]:
        """
        Load all data tables.
        
        Returns:
            Dictionary containing all DataFrames
        """
        return {
            'introductory': self.load_introductory(),
            'home_training': self.load_home_training(),
            'intensive_therapies': self.load_intensive_therapies(),
            'motorical_development': self.load_motorical_development()
        }


# Connects to database and loads all data
def load_data(conn: str) -> dict[str, pl.DataFrame]:
    db = SurveyDatabase(conn)
    return db.load_all_data()


if __name__ == "__main__":
    conn = get_connection()  
    
    data = load_data(conn)
    
    for table_name, df in data.items():
        print(f"\n{table_name}: {df.shape}")
        print(df.head(0))