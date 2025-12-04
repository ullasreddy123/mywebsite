import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    # Connect to default postgres database first
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="ullas2004",
        host="localhost"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Create database if it doesn't exist
    try:
        cur.execute("CREATE DATABASE votingdb")
        print("Database 'votingdb' created successfully!")
    except psycopg2.Error as e:
        print(f"Database may already exist: {e}")
    finally:
        cur.close()
        conn.close()

    # Connect to our new database
    conn = psycopg2.connect(
        dbname="votingdb",
        user="postgres",
        password="ullas2004",
        host="localhost"
    )
    cur = conn.cursor()

    try:
        # Create voters table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS voters (
                id SERIAL PRIMARY KEY,
                gmail VARCHAR(255) UNIQUE NOT NULL,
                voter_id VARCHAR(255) UNIQUE NOT NULL,
                face_code VARCHAR(255) NOT NULL
            )
        """)
        print("Voters table created successfully!")

        # Create candidates table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL
            )
        """)
        print("Candidates table created successfully!")

        # Create votes table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                id SERIAL PRIMARY KEY,
                voter_id VARCHAR(255) REFERENCES voters(voter_id),
                candidate_id INTEGER REFERENCES candidates(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(voter_id)
            )
        """)
        print("Votes table created successfully!")

        # Commit the changes
        conn.commit()
        print("All tables created successfully!")

    except psycopg2.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    setup_database()