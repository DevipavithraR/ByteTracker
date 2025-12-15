from utils.db_conn import get_db_connection

try:
    conn = get_db_connection()
    print("✅ MySQL connected successfully")
    conn.close()
except Exception as e:
    print("❌ Error:", e)
