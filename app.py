
from flask import Flask, render_template, request, redirect, session, flash, send_from_directory
from functools import wraps
import uuid
import psycopg2
import cv2
import os
import json
import numpy as np
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except Exception:
    HAS_FACE_RECOGNITION = False
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from typing import Optional
import re

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # Set this properly in production

def get_db_connection():
    return psycopg2.connect(
        dbname=os.environ.get('DB_NAME', 'votingdb'),
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'ullas2004'),
        host=os.environ.get('DB_HOST', 'localhost')
    
    )

def ensure_db_migration():
    """Ensure database has the `face_image` column on voters table. Run at startup."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Add column if not exists (Postgres supports IF NOT EXISTS)
        cur.execute("ALTER TABLE voters ADD COLUMN IF NOT EXISTS face_image TEXT")
        # Add column to store face embedding (JSON text of list of floats)
        cur.execute("ALTER TABLE voters ADD COLUMN IF NOT EXISTS face_encoding TEXT")
        conn.commit()
    except Exception:
        # Migration is best-effort; if it fails we continue and code will fallback to face_code
        pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def _image_histogram_similarity(path_a: str, path_b: str) -> float:
    """Return histogram correlation between two images (1.0 best)."""
    try:
        a = cv2.imread(path_a)
        b = cv2.imread(path_b)
        if a is None or b is None:
            return 0.0
        # Resize to fixed size for consistency
        a = cv2.resize(a, (200, 200))
        b = cv2.resize(b, (200, 200))
        # Convert to HSV and compute color histograms
        hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        # score range: -1..1, convert to 0..1
        return max(0.0, min(1.0, (score + 1) / 2))
    except Exception:
        return 0.0

def compute_face_encoding(image_path: str) -> Optional[list]:
    """Return a 128-d face embedding list for the first face found in image, or None."""
    try:
        if HAS_FACE_RECOGNITION:
            img = face_recognition.load_image_file(image_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                # convert numpy array to regular list for JSON storage
                return encs[0].tolist()
            return None
        else:
            # face_recognition not available -> fallback: try to detect face using OpenCV Haar cascades is non-trivial
            # We'll return None so the code falls back to histogram matching
            return None
    except Exception:
        return None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin' not in session:
            flash('Please log in as admin first.', 'error')
            return redirect('/admin/login')
        return f(*args, **kwargs)
    return decorated_function


def is_valid_voter_id(voter_id: str) -> bool:
    """Validate voter ID: exactly 10 chars, first 3 letters then 7 digits."""
    if not isinstance(voter_id, str):
        return False
    voter_id = voter_id.strip()
    # Regex: 3 letters (A-Z or a-z) followed by 7 digits
    return bool(re.fullmatch(r"[A-Za-z]{3}\d{7}", voter_id))


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Simple admin login. Password is read from ADMIN_PASSWORD env var (default: admin123).
    This is intentionally minimal — for production use a proper auth system.
    """
    if request.method == 'POST':
        password = request.form.get('password', '')
        admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
        if password == admin_password:
            session['admin'] = True
            flash('Admin logged in successfully', 'success')
            return redirect('/admin')
        else:
            flash('Invalid admin password', 'error')
            return redirect('/admin/login')

    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Logged out', 'success')
    return redirect('/')

@app.route('/results')
def view_results():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get voting results
        cur.execute("""
            SELECT c.name, c.image, COUNT(v.id) as vote_count
            FROM candidates c
            LEFT JOIN votes v ON c.id = v.candidate_id
            GROUP BY c.id, c.name, c.image
            ORDER BY vote_count DESC
        """)
        results = cur.fetchall()
        
        # Calculate total votes
        total_votes = sum(result[2] for result in results)
        
        # Format results with percentages
        formatted_results = []
        for name, image, votes in results:
            percentage = (votes / total_votes * 100) if total_votes > 0 else 0
            formatted_results.append({
                'name': name,
                'image': image,
                'votes': votes,
                'percentage': percentage
            })
        
        # Identify the leading candidate
        leading_candidate = formatted_results[0] if formatted_results else None
        
        return render_template('results.html', 
                            results=formatted_results,
                            total_votes=total_votes,
                            leading_candidate=leading_candidate)
    except Exception as e:
        flash(f'Error retrieving results: {str(e)}')
        return redirect('/')
    finally:
        cur.close()
        conn.close()

@app.route('/')
def index():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get voting results
        cur.execute("""
            SELECT c.name, COUNT(v.id) as vote_count 
            FROM candidates c 
            LEFT JOIN votes v ON c.id = v.candidate_id 
            GROUP BY c.id, c.name 
            ORDER BY vote_count DESC
        """)
        results = cur.fetchall()
        
        # Calculate total votes
        total_votes = sum(r[1] for r in results)
        
        return render_template('index.html', results=results, total_votes=total_votes)
    except Exception as e:
        # If there's an error, just show the page without statistics
        return render_template('index.html',
                            total_voters=0,
                            total_votes=0,
                            total_candidates=0)
    finally:
        if 'cur' in locals() and cur is not None:
            cur.close()
        if 'conn' in locals() and conn is not None:
            conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            voter_id = request.form['voter_id']
            # Validate voter ID format before attempting login
            if not is_valid_voter_id(voter_id):
                flash('Invalid Voter ID format. It must be 3 letters followed by 7 digits.', 'error')
                return redirect('/login')
            gmail = request.form['gmail']
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Verify voter credentials
            cur.execute("SELECT id FROM voters WHERE voter_id = %s AND gmail = %s", (voter_id, gmail))
            voter = cur.fetchone()
            
            if voter:
                session['voter_id'] = voter_id
                session['voter_authenticated'] = True
                flash('Login successful!', 'success')
                return redirect('/vote')
            else:
                flash('Invalid voter ID or email', 'error')
                return redirect('/login')
                
        except Exception as e:
            flash(f'Login failed: {str(e)}', 'error')
            return redirect('/login')
        finally:
            cur.close()
            conn.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()  # Clear all session data including face verification
    flash('Logged out successfully', 'success')
    return redirect('/')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            gmail = request.form['gmail']
            voter_id = request.form['voter_id']
            # Validate voter ID format: 3 letters followed by 7 digits (total 10 characters)
            if not is_valid_voter_id(voter_id):
                flash('Voter ID must be 3 letters followed by 7 digits (example: ABC1234567).', 'error')
                return redirect('/register')
            # Prefer a captured face image stored in session (filename) or an uploaded file
            face_image = session.get('face_image') or ''

            # Handle optional uploaded face image from form input 'face_upload'
            if 'face_upload' in request.files:
                upload = request.files['face_upload']
                if upload and upload.filename:
                    img_dir = os.path.join(os.path.dirname(__file__), 'face_captures')
                    os.makedirs(img_dir, exist_ok=True)
                    filename = secure_filename(upload.filename)
                    # avoid collisions
                    filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                    upload.save(os.path.join(img_dir, filename))
                    face_image = filename
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Check if voter already exists
            cur.execute("SELECT * FROM voters WHERE voter_id = %s OR gmail = %s", (voter_id, gmail))
            if cur.fetchone():
                flash('Voter ID or Gmail already registered', 'error')
                return redirect('/register')
                
            # Ensure the captured image file exists before saving
            saved_face = None
            if face_image:
                img_dir = os.path.join(os.path.dirname(__file__), 'face_captures')
                img_path = os.path.join(img_dir, face_image)
                if os.path.exists(img_path):
                    saved_face = face_image

            # Register voter with face authentication (store captured image filename in face_image column)
            face_encoding_json = None
            if saved_face is not None:
                # try to compute face embedding (if face_recognition is available)
                try:
                    img_path = os.path.join(os.path.dirname(__file__), 'face_captures', saved_face)
                    encoding = compute_face_encoding(img_path)
                    if encoding:
                        face_encoding_json = json.dumps(encoding)
                except Exception:
                    face_encoding_json = None

            if saved_face is not None and face_encoding_json is not None:
                cur.execute("INSERT INTO voters (gmail, voter_id, face_image, face_encoding) VALUES (%s, %s, %s, %s)", 
                           (gmail, voter_id, saved_face, face_encoding_json))
            elif saved_face is not None:
                cur.execute("INSERT INTO voters (gmail, voter_id, face_image) VALUES (%s, %s, %s)", 
                           (gmail, voter_id, saved_face))
            else:
                # fallback to inserting without face_image
                cur.execute("INSERT INTO voters (gmail, voter_id) VALUES (%s, %s)", 
                           (gmail, voter_id))
            conn.commit()
            flash('Registration successful! Please login to continue.', 'success')
            # Clear temporary session capture after saving to DB
            session.pop('face_image', None)
            session.pop('face_captured', None)
            return redirect('/login')
        except Exception as e:
            flash(f'Registration failed: {str(e)}', 'error')
            return redirect('/register')
        finally:
            cur.close()
            conn.close()
    return render_template('register.html')

def voter_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('voter_authenticated'):
            flash('Please login first.', 'error')
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/vote', methods=['GET', 'POST'])
@voter_required
def vote():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if request.method == 'POST':
            if not session.get('face_verified'):
                flash('Please complete face verification before voting', 'error')
                return redirect('/vote')
                
            voter_id = session.get('voter_id')
            candidate_id = request.form.get('candidate')
            
            if not candidate_id:
                flash('No candidate selected')
                return redirect('/vote')
                
            # Check if voter has already voted
            cur.execute("SELECT * FROM votes WHERE voter_id = %s", (voter_id,))
            if cur.fetchone():
                flash('You have already voted')
                return redirect('/vote')
                
            # Insert the vote
            cur.execute("INSERT INTO votes (voter_id, candidate_id) VALUES (%s, %s)", 
                       (voter_id, candidate_id))
            conn.commit()

            # Get voting results
            cur.execute("""
                SELECT c.name, COUNT(v.id) as vote_count
                FROM candidates c
                LEFT JOIN votes v ON c.id = v.candidate_id
                GROUP BY c.id, c.name
                ORDER BY vote_count DESC
            """)
            results = cur.fetchall()
            
            # Calculate total votes
            total_votes = sum(result[1] for result in results)
            
            # Format results with percentages
            formatted_results = []
            for name, votes in results:
                percentage = (votes / total_votes * 100) if total_votes > 0 else 0
                formatted_results.append({
                    'name': name,
                    'votes': votes,
                    'percentage': percentage
                })
            
            return render_template('results.html', 
                                results=formatted_results,
                                total_votes=total_votes)
            
        cur.execute("SELECT * FROM candidates")
        candidates = cur.fetchall()
        return render_template('vote.html', candidates=candidates)
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect('/')
    finally:
        cur.close()
        conn.close()

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Ensure candidates table has an image column
        try:
            cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS image VARCHAR(255)")
            conn.commit()
        except Exception:
            # ignore if cannot alter (older PG versions) and continue
            conn.rollback()
        
        if request.method == 'POST':
            if 'add' in request.form:
                name = request.form['name']
                # handle optional image upload
                image_filename = None
                if 'image' in request.files:
                    img = request.files['image']
                    if img and img.filename:
                        upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'images', 'candidates')
                        os.makedirs(upload_dir, exist_ok=True)
                        filename = secure_filename(img.filename)
                        # avoid collisions by prefixing timestamp
                        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                        img.save(os.path.join(upload_dir, filename))
                        image_filename = filename

                if image_filename:
                    cur.execute("INSERT INTO candidates (name, image) VALUES (%s, %s)", (name, image_filename))
                else:
                    cur.execute("INSERT INTO candidates (name) VALUES (%s)", (name,))
                flash('Candidate added successfully!', 'success')
            elif 'remove' in request.form:
                cid = request.form['cid']
                # fetch image filename to delete file
                cur.execute("SELECT image FROM candidates WHERE id = %s", (cid,))
                row = cur.fetchone()
                if row and row[0]:
                    try:
                        img_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'candidates', row[0])
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except Exception:
                        pass
                cur.execute("DELETE FROM candidates WHERE id = %s", (cid,))
                flash('Candidate removed successfully!', 'success')
            elif 'clear_votes' in request.form:
                # Erase all votes
                try:
                    cur.execute("DELETE FROM votes")
                    conn.commit()
                    flash('All votes have been erased.', 'success')
                except Exception as e:
                    conn.rollback()
                    flash(f'Failed to erase votes: {str(e)}', 'error')
            conn.commit()
            
        # Fetch voters
        cur.execute("SELECT id, gmail, voter_id FROM voters ORDER BY id")
        voters = cur.fetchall()
        
        # Fetch candidates
        cur.execute("SELECT * FROM candidates")
        candidates = cur.fetchall()
        
        # If delete_voter action is submitted
        if 'delete_voter' in request.form:
            voter_id = request.form['voter_id']
            try:
                # First delete any votes by this voter
                cur.execute("DELETE FROM votes WHERE voter_id = %s", (voter_id,))
                # Then delete the voter
                cur.execute("DELETE FROM voters WHERE voter_id = %s", (voter_id,))
                conn.commit()
                flash('Voter deleted successfully!', 'success')
            except Exception as e:
                conn.rollback()
                flash(f'Failed to delete voter: {str(e)}', 'error')
            return redirect('/admin')
            
        return render_template('dashboard.html', candidates=candidates, voters=voters)
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect('/admin')
    finally:
        cur.close()
        conn.close()
@app.route('/face_image/<path:filename>')
def face_image(filename):
    directory = os.path.join(os.path.dirname(__file__), 'face_captures')
    return send_from_directory(directory, filename)

@app.route('/face_auth')
def face_auth():
    try:
        # Allow capturing face both during registration (unauthenticated) and during voting (authenticated)
        cam = cv2.VideoCapture(0)
        # choose redirect target depending on whether user is logged-in
        redirect_target = '/vote' if session.get('voter_authenticated') else '/register'

        if not cam.isOpened():
            flash('Could not access the camera', 'error')
            return redirect(redirect_target)

        cv2.namedWindow("Face Authentication")
        img_dir = os.path.join(os.path.dirname(__file__), 'face_captures')
        os.makedirs(img_dir, exist_ok=True)

        # Prepare a Haar cascade as a fallback face detector when face_recognition is not available
        face_cascade = None
        try:
            if not HAS_FACE_RECOGNITION:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            face_cascade = None

        while True:
            ret, frame = cam.read()
            if not ret:
                flash('Failed to capture image', 'error')
                break
            cv2.imshow("Face Authentication", frame)

            k = cv2.waitKey(1)
            if k % 256 == 32:  # Press SPACE to capture
                # Before saving, ensure a face is actually detected to avoid false accepts
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    rgb = frame

                detected_face = False
                single_face = False
                captured_encoding = None

                if HAS_FACE_RECOGNITION:
                    try:
                        # Use face_recognition to find faces and encodings
                        face_locs = face_recognition.face_locations(rgb)
                        if len(face_locs) == 1:
                            single_face = True
                            encs = face_recognition.face_encodings(rgb, face_locs)
                            if encs:
                                captured_encoding = encs[0]
                                detected_face = True
                        elif len(face_locs) > 1:
                            flash('Multiple faces detected. Ensure only your face is visible.', 'error')
                            detected_face = False
                        else:
                            flash('No face detected. Please position your face in front of the camera.', 'error')
                            detected_face = False
                    except Exception as e:
                        flash(f'Face detection error: {str(e)}', 'error')
                        detected_face = False
                else:
                    # Fallback: use OpenCV Haar cascade if available
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if face_cascade is not None:
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                            if len(faces) == 1:
                                detected_face = True
                                single_face = True
                            elif len(faces) > 1:
                                flash('Multiple faces detected. Ensure only your face is visible.', 'error')
                                detected_face = False
                            else:
                                flash('No face detected. Please position your face in front of the camera.', 'error')
                                detected_face = False
                        else:
                            # If no detector available, conservatively reject capture
                            flash('No face detector available on this system. Install face_recognition or ensure OpenCV has haarcascades.', 'error')
                            detected_face = False
                    except Exception as e:
                        flash(f'Face detection error: {str(e)}', 'error')
                        detected_face = False

                if not detected_face:
                    # Do not accept this capture; continue capturing loop
                    continue

                # Use a UUID-based filename to avoid collisions and unsafe chars
                img_filename = f'face_capture_{uuid.uuid4().hex}.png'
                img_path = os.path.join(img_dir, img_filename)
                cv2.imwrite(img_path, frame)

                # Save the captured image filename in session
                session['face_image'] = img_filename
                session['face_captured'] = True

                # If user is logged in, verify they have a stored face in DB
                if session.get('voter_authenticated'):
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute("SELECT face_image, face_encoding FROM voters WHERE voter_id = %s", (session['voter_id'],))
                        row = cur.fetchone()
                    finally:
                        if 'cur' in locals() and cur is not None:
                            cur.close()
                        if 'conn' in locals() and conn is not None:
                            conn.close()

                    if row:
                        stored_filename = row[0]
                        stored_encoding_json = row[1] if len(row) > 1 else None

                        verified = False

                        # Prefer embedding-based comparison when possible
                        if stored_encoding_json and HAS_FACE_RECOGNITION and captured_encoding is not None:
                            try:
                                stored_encoding = np.array(json.loads(stored_encoding_json))
                                # captured_encoding is already a numpy array
                                try:
                                    dist = face_recognition.face_distance([stored_encoding], captured_encoding)[0]
                                except Exception:
                                    dist = float(np.linalg.norm(stored_encoding - captured_encoding))
                                # Use a stricter threshold (e.g., 0.55) to reduce false accepts
                                if dist <= 0.55:
                                    verified = True
                            except Exception:
                                verified = False

                        # Fallback: histogram match on images (raise threshold)
                        if not verified and stored_filename:
                            stored_path = os.path.join(img_dir, stored_filename)
                            try:
                                similarity = _image_histogram_similarity(stored_path, img_path)
                                # increase similarity threshold to reduce false positives
                                if similarity >= 0.6:
                                    verified = True
                            except Exception:
                                verified = False

                        if verified:
                            session['face_verified'] = True
                            flash('Face verification successful', 'success')
                        else:
                            session['face_verified'] = False
                            flash('Face did not match our records. Verification failed.', 'error')
                    else:
                        session['face_verified'] = False
                        flash('No stored face found for this voter. Please register with face first.', 'error')
                else:
                    # Not logged in: this capture is for registration. Keep it in session so register can use it.
                    flash('Face captured successfully. Continue registration to save it.', 'success')

                break
            elif k % 256 == 27:  # ESC to exit
                break

        cam.release()
        cv2.destroyAllWindows()
        return redirect(redirect_target)
    except Exception as e:
        flash(f'Face authentication error: {str(e)}', 'error')
        # fallback redirect
        return redirect('/register')
if __name__ == '__main__':
    # Ensure DB has needed columns before starting
    try:
        ensure_db_migration()
    except Exception:
        pass
    app.run(debug=True)