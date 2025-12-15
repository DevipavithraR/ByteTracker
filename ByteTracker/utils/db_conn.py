import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="pythonuser",
        password="pythonpass",
        database="attendance_details"
    )
