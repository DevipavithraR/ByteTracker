# import mysql.connector

# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="pythonuser",
#         password="pythonpass",
#         database="attendance_details"
#     )

import psycopg2

def get_db_connection():    
    return psycopg2.connect(
        host="localhost",
        database="bytetrakcer_attendance_db",
        user="postgres",
        password="Test@123",
        port=5432
    )
