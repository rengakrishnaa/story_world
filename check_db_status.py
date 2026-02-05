
import sqlite3
import json

def check_status():
    conn = sqlite3.connect("local.db")
    cursor = conn.cursor()
    
    ep_id = "49e76d44-292f-4b0e-b158-294979963908"
    print(f"Checking Episode: {ep_id}")
    
    cursor.execute("SELECT beat_id, state, last_error FROM beats WHERE episode_id = ?", (ep_id,))
    rows = cursor.fetchall()
    
    all_accepted = True
    for r in rows:
        print(f"  - Beat {r[0]}:\n      State: {r[1]}\n      Error: {r[2]}")
        if r[1] != "ACCEPTED":
            all_accepted = False
            
    if all_accepted and rows:
        print("\n✅ VERIFIED: Episode is fully ACCEPTED!")
    else:
        print("\n❌ STILL STUCK: One or more beats are not ACCEPTED.")

if __name__ == "__main__":
    check_status()
