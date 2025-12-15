from services.attendance_service import attendance_service

print("\n💡 Press 'q' at any time to stop the script.")
print("💡 Or create stop.flag to stop.")
print("💡 Or press CTRL + C.\n")

# --------------------------------------------
try:
    attendance_service()
    # break_service()
    # logout_service()

# ---- Catch Ctrl+C ----
except KeyboardInterrupt:
    print("\n🛑 CTRL + C detected — script stopped safely.")

except SystemExit:
    print("🛑 Pipeline stopped gracefully.")

except Exception as e:
    print(f"❌ Unexpected Error: {e}")
