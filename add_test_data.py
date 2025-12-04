import psycopg2
from datetime import datetime

def add_test_data():
    # Connect to the database
    conn = psycopg2.connect(
        dbname="votingdb",
        user="postgres",
        password="ullas2004",
        host="localhost"
    )
    cur = conn.cursor()

    try:
        # First, clear existing test data
        print("Clearing existing data...")
        cur.execute("DELETE FROM votes")
        cur.execute("DELETE FROM voters")
        cur.execute("DELETE FROM candidates")
        
        # Add test candidates
        candidates = [
            "John Smith",
            "Emma Wilson",
            "Michael Brown",
            "Sarah Davis",
            "James Johnson"
        ]
        
        print("\nAdding candidates:")
        for candidate in candidates:
            cur.execute(
                "INSERT INTO candidates (name) VALUES (%s) RETURNING id",
                (candidate,)
            )
            candidate_id = cur.fetchone()[0]
            print(f"Added candidate: {candidate} (ID: {candidate_id})")

        # Add test voters
        voters = [
            ("voter1@example.com", "VOTER001", "face_code_001"),
            ("voter2@example.com", "VOTER002", "face_code_002"),
            ("voter3@example.com", "VOTER003", "face_code_003"),
            ("voter4@example.com", "VOTER004", "face_code_004"),
            ("voter5@example.com", "VOTER005", "face_code_005")
        ]
        
        print("\nAdding voters:")
        for gmail, voter_id, face_code in voters:
            cur.execute(
                "INSERT INTO voters (gmail, voter_id, face_code) VALUES (%s, %s, %s)",
                (gmail, voter_id, face_code)
            )
            print(f"Added voter: {voter_id} ({gmail})")

        # Add some test votes
        # First, get the actual candidate IDs
        cur.execute("SELECT id, name FROM candidates ORDER BY id LIMIT 2")
        candidate_ids = cur.fetchall()
        
        votes = [
            ("VOTER001", candidate_ids[0][0]),  # Voter1 votes for first candidate
            ("VOTER002", candidate_ids[1][0]),  # Voter2 votes for second candidate
            ("VOTER003", candidate_ids[0][0])   # Voter3 votes for first candidate
        ]
        
        print("\nAdding test votes:")
        for voter_id, candidate_id in votes:
            cur.execute(
                "INSERT INTO votes (voter_id, candidate_id) VALUES (%s, %s)",
                (voter_id, candidate_id)
            )
            # Get candidate name for the message
            cur.execute("SELECT name FROM candidates WHERE id = %s", (candidate_id,))
            candidate_name = cur.fetchone()[0]
            print(f"Added vote: Voter {voter_id} voted for {candidate_name}")

        # Commit the changes
        conn.commit()
        print("\nAll test data added successfully!")

        # Display current standings
        print("\nCurrent Vote Count:")
        cur.execute("""
            SELECT c.name, COUNT(v.id) as vote_count 
            FROM candidates c 
            LEFT JOIN votes v ON c.id = v.candidate_id 
            GROUP BY c.id, c.name 
            ORDER BY vote_count DESC
        """)
        results = cur.fetchall()
        for name, count in results:
            print(f"{name}: {count} votes")

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    add_test_data()