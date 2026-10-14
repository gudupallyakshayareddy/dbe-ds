
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    send_from_directory,
    jsonify
)

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import psycopg2
from pgvector.psycopg2 import register_vector



import os
import numpy as np
import re
import hashlib
import uuid
from werkzeug.utils import secure_filename


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "research_repository_secret_key"


# ============================================================
# FLASK LOGIN
# ============================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_document_embedding(text, chunk_size=180, overlap=40):
    """Represent a long document using overlapping text chunks."""
    import re
    import numpy as np

    if not text or not text.strip():
        return None

    # Normalize whitespace before splitting into words.
    words = re.sub(r"\s+", " ", text).strip().split()

    if not words:
        return None

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)

    # Encode each chunk rather than truncating the whole document.
    vectors = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Combine chunk vectors into one vector per paper.
    document_vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(document_vector)

    if norm > 0:
        document_vector = document_vector / norm

    return np.asarray(document_vector, dtype=np.float32)



print("Embedding model loaded!")


def preprocess_pdf_text(text):
    """Clean extracted PDF text before storing or embedding it."""
    if not text:
        return ""

    # Normalize whitespace and remove null characters.
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove repeated whitespace while preserving paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# POSTGRESQL DATABASE CONFIGURATION
# ============================================================

DB_HOST = "localhost"

DB_NAME = "research_repository"

DB_USER = "postgres"

DB_PASSWORD = "Harshini@2006"

DB_PORT = "5432"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = psycopg2.connect(

        host=DB_HOST,

        database=DB_NAME,

        user=DB_USER,

        password=DB_PASSWORD,

        port=DB_PORT

    )

    register_vector(connection)

    return connection


# ============================================================
# USER CLASS
# ============================================================

class User(UserMixin):

    def __init__(
        self,
        user_id,
        username
    ):

        self.id = user_id

        self.username = username


# ============================================================
# LOAD USER
# ============================================================

@login_manager.user_loader
def load_user(user_id):

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                username
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )

        user = cursor.fetchone()

        cursor.close()

        connection.close()

        if user:

            return User(
                user[0],
                user[1]
            )

    except Exception as e:

        print(
            "User loading error:",
            e
        )

    return None


# ============================================================
# HOME / DOCUMENT UPLOAD
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
@login_required
def upload_file():

    extracted_text = ""

    filename = ""

    # ========================================================
    # GET REQUEST
    # ========================================================

    if request.method == "GET":

        return render_template(

            "index.html",

            filename="",

            extracted_text=""

        )

    # ========================================================
    # GET UPLOADED FILE
    # ========================================================

    file = request.files.get("file")

    if not file or file.filename == "":

        return render_template(

            "index.html",

            filename="",

            extracted_text=""

        )

    # ========================================================
    # CHECK PDF
    # ========================================================

    if not file.filename.lower().endswith(".pdf"):

        return render_template(

            "index.html",

            filename="",

            extracted_text=""

        )

    # ========================================================
    # ORIGINAL FILE NAME
    # ========================================================

    original_filename = secure_filename(
        file.filename
    )

    # ========================================================
    # CREATE UNIQUE SERVER FILE NAME
    #
    # This prevents two users from overwriting files
    # when they upload PDFs with the same filename.
    # ========================================================

    unique_filename = (
        str(current_user.id)
        + "_"
        + str(uuid.uuid4())
        + "_"
        + original_filename
    )

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        unique_filename

    )

    file.save(filepath)

    print(
        "PDF uploaded successfully:",
        original_filename
    )

    # ========================================================
    # TEXT EXTRACTION
    # ========================================================

    try:

        reader = PdfReader(filepath)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                extracted_text += (
                    page_text + "\n"
                )

        print(
            "Text extracted successfully."
        )

    except Exception as e:

        print(
            "PDF extraction error:",
            e
        )

        if os.path.exists(filepath):

            os.remove(filepath)

        return render_template(

            "index.html",

            filename=original_filename,

            extracted_text=""

        )


     
    # ========================================================
    # TEXT PREPROCESSING AND VALIDATION
    # ========================================================


    extracted_text = preprocess_pdf_text(extracted_text)

    if len(extracted_text) < 30:
        if os.path.exists(filepath):
            os.remove(filepath)

        return render_template(
            "index.html",
            filename=original_filename,
            extracted_text="",
            error="This PDF contains too little extractable text."
        )
    # ========================================================
    # EMBEDDING
    # ========================================================

    embedding = None

    if extracted_text.strip():

        embedding = generate_document_embedding(
    extracted_text
)

        print(
            "Embedding generated successfully!"
        )

    else:

        print(
            "No text found in PDF."
        )

    # ========================================================
    # DATABASE STORAGE + AUDIT LOGGING
    # ========================================================

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        title = os.path.splitext(original_filename)[0]

        # Insert paper and retrieve its generated ID
        cursor.execute(
            """
            INSERT INTO research_papers
            (
                user_id,
                title,
                filename,
                extracted_text,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                current_user.id,
                title,
                unique_filename,
                extracted_text,
                embedding
            )
        )

        paper_id = cursor.fetchone()[0]

        # Record the upload in the audit table
        cursor.execute(
            """
            INSERT INTO research_audit_log
            (
                paper_id,
                user_id,
                action,
                details
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                paper_id,
                current_user.id,
                "INSERT",
                "Research paper uploaded"
            )
        )

        # Save both changes together
        connection.commit()

        print(
            "Paper stored for user:",
            current_user.username
        )

    except Exception as e:
        if connection is not None:
            connection.rollback()

        print("Database error:", e)

        if os.path.exists(filepath):
            os.remove(filepath)

        return render_template(
            "index.html",
            filename="",
            extracted_text=""
        )

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    # ========================================================
    # DISPLAY ORIGINAL NAME
    # ========================================================

    return render_template(
        "index.html",
        filename=original_filename,
        extracted_text=extracted_text
    )

# ============================================================
# SEARCH
# ============================================================

@app.route(
    "/search",
    methods=["GET", "POST"]
)
@login_required
def search():

    query = ""

    results = []

    # ========================================================
    # POST SEARCH
    # ========================================================

    if request.method == "POST":

        query = request.form.get(
            "query",
            ""
        ).strip()

        if query:

            # =================================================
            # QUERY EMBEDDING
            # =================================================

            query_embedding = embedding_model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            query_embedding = np.asarray(
                query_embedding,
                dtype=np.float32
            )
            print("Search query:", repr(query))
            print("Query embedding shape:", query_embedding.shape)
            print("Query embedding finite:", np.isfinite(query_embedding).all())
            print("Query embedding norm:", np.linalg.norm(query_embedding))
            
            print(
                "Query embedding generated!"
            )

            # =================================================
            # USER-SPECIFIC SEMANTIC SEARCH
            # =================================================

            try:

                connection = get_db_connection()

                cursor = connection.cursor()

                search_query = """

                    SELECT

                        id,

                        title,

                        filename,

                        extracted_text,

                        1 - (
                            embedding <=> %s
                        ) AS similarity

                    FROM research_papers

                    WHERE
                        user_id = %s

                    AND
                        embedding IS NOT NULL

                    ORDER BY
                        embedding <=> %s

                    LIMIT 10

                """

                cursor.execute(

                    search_query,

                    (
                        query_embedding,
                        current_user.id,
                        query_embedding
                    )

                )

                results = cursor.fetchall()

                results = sorted(

                    results,

                    key=lambda x: x[4],

                    reverse=True

                )

                cursor.close()

                connection.close()

                print(
                    "Search completed for:",
                    current_user.username
                )

                print(
                    "Results found:",
                    len(results)
                )

            except Exception as e:

                print(
                    "Search database error:",
                    e
                )

    # ========================================================
    # IMPORTANT:
    #
    # Search has its OWN PAGE.
    # ========================================================

    return render_template(

        "search.html",

        query=query,

        results=results

    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("upload_file")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            return render_template(

                "register.html",

                error=(
                    "Please enter username "
                    "and password."
                )

            )

        if len(username) < 3:

            return render_template(

                "register.html",

                error=(
                    "Username must contain "
                    "at least 3 characters."
                )

            )

        if len(password) < 6:

            return render_template(

                "register.html",

                error=(
                    "Password must contain "
                    "at least 6 characters."
                )

            )

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            cursor.execute(

                """
                SELECT id
                FROM users
                WHERE username = %s
                """,

                (username,)

            )

            existing_user = cursor.fetchone()

            if existing_user:

                cursor.close()

                connection.close()

                return render_template(

                    "register.html",

                    error=(
                        "Username already exists."
                    )

                )

            hashed_password = generate_password_hash(
                password
            )

            cursor.execute(

                """
                INSERT INTO users
                (
                    username,
                    password
                )
                VALUES
                (
                    %s,
                    %s
                )
                """,

                (
                    username,
                    hashed_password
                )

            )

            connection.commit()

            cursor.close()

            connection.close()

            print(
                "User registered:",
                username
            )

            return redirect(
                url_for("login")
            )

        except Exception as e:

            print(
                "Registration error:",
                e
            )

            return render_template(

                "register.html",

                error=(
                    "Registration failed. "
                    "Please try again."
                )

            )

    return render_template(
        "register.html"
    )



# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    # If the user is already logged in
    if current_user.is_authenticated:

        next_page = request.args.get("next")

        if (
            next_page
            and next_page.startswith("/")
            and not next_page.startswith("//")
        ):
            return redirect(next_page)

        return redirect(url_for("upload_file"))

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT
                    id,
                    username,
                    password
                FROM users
                WHERE username = %s
                """,
                (username,)
            )

            user = cursor.fetchone()

            cursor.close()
            connection.close()

            if user and check_password_hash(
                user[2],
                password
            ):

                logged_in_user = User(
                    user[0],
                    user[1]
                )

                login_user(logged_in_user)

                print(
                    "Login successful:",
                    username
                )

                # Return to the page the user originally requested
                next_page = request.args.get("next")

                if (
                    next_page
                    and next_page.startswith("/")
                    and not next_page.startswith("//")
                ):
                    return redirect(next_page)

                # Default destination when no page was requested
                return redirect(
                    url_for("upload_file")
                )

            return render_template(
                "login.html",
                error="Invalid username or password."
            )

        except Exception as e:

            print(
                "Login error:",
                e
            )

            return render_template(
                "login.html",
                error="Login failed. Please try again."
            )

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    username = current_user.username

    logout_user()

    print(
        "User logged out:",
        username
    )

    return redirect(
        url_for("login")
    )



# ============================================================
# REPOSITORY
# ============================================================

@app.route("/repository")
@login_required
def repository():
    papers = []
    error_message = None

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                title,
                filename,
                uploaded_at
            FROM research_papers
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            """,
            (current_user.id,)
        )

        papers = cursor.fetchall()

        print("Repository loaded for:", current_user.username)
        print("Papers found:", len(papers))

    except Exception as e:
        print("Repository database error:", e)
        error_message = (
            "Unable to load your papers. Please try again."
        )

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "repository.html",
        papers=papers,
        error_message=error_message
    )

# ============================================================
# RESEARCH PAPER ANALYSIS
# ============================================================

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def analyze_paper_comparison(text1, text2):
    """Compare paper text and return evidence-based observations."""

    def split_sentences(text):
        text = re.sub(r"\s+", " ", text or "").strip()
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if len(sentence.strip()) >= 35
        ]

    def find_evidence(sentences, terms, limit=3):
        matches = [
            sentence for sentence in sentences
            if any(term in sentence.lower() for term in terms)
        ]
        return matches[:limit]

    sentences1 = split_sentences(text1)
    sentences2 = split_sentences(text2)

    if not text1 or not text1.strip() or not text2 or not text2.strip():
        return {
            "error": "One or both papers do not contain enough extracted text.",
            "similarity": None,
            "common_topics": [],
            "unique_topics_1": [],
            "unique_topics_2": [],
            "objectives": [[], []],
            "methods": [[], []],
            "findings": [[], []],
            "limitations": [[], []],
            "gaps": []
        }

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1500,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    try:
        matrix = vectorizer.fit_transform([text1, text2])
    except ValueError:
        return {
            "error": "Not enough readable text to compare these papers.",
            "similarity": None,
            "common_topics": [],
            "unique_topics_1": [],
            "unique_topics_2": [],
            "objectives": [[], []],
            "methods": [[], []],
            "findings": [[], []],
            "limitations": [[], []],
            "gaps": []
        }

    terms = vectorizer.get_feature_names_out()
    scores1 = matrix[0].toarray().ravel()
    scores2 = matrix[1].toarray().ravel()

    ranked1 = sorted(
        zip(terms, scores1), key=lambda item: item[1], reverse=True
    )
    ranked2 = sorted(
        zip(terms, scores2), key=lambda item: item[1], reverse=True
    )

   
    top1 = [term for term, score in ranked1 if score > 0][:15]
    top2 = [term for term, score in ranked2 if score > 0][:15]

    common = [
        term for term in top1 if term in set(top2)
    ][:10]

    unique1 = [term for term in top1 if term not in set(top2)][:8]
    unique2 = [term for term in top2 if term not in set(top1)][:8]

    objective_terms = [
        "objective", "aim", "purpose", "this study", "we propose",
        "we investigate", "we present", "this paper"
    ]
    method_terms = [
        "method", "methodology", "proposed approach", "algorithm",
        "framework", "dataset", "we use", "model was", "we trained"
    ]
    finding_terms = [
        "results show", "results indicate", "we found",
        "outperformed", "improved", "achieved", "demonstrate",
        "performance", "accuracy"
    ]
    limitation_terms = [
        "limitation", "limited", "future work", "however",
        "remains a challenge", "further research", "not address",
        "lack of", "small sample", "in the future", "shortcoming"
    ]

    objectives = [
        find_evidence(sentences1, objective_terms),
        find_evidence(sentences2, objective_terms)
    ]
    methods = [
        find_evidence(sentences1, method_terms),
        find_evidence(sentences2, method_terms)
    ]
    findings = [
        find_evidence(sentences1, finding_terms),
        find_evidence(sentences2, finding_terms)
    ]
    limitations = [
        find_evidence(sentences1, limitation_terms),
        find_evidence(sentences2, limitation_terms)
    ]

    gaps = []
    for paper_number, evidence in enumerate(limitations, start=1):
        for sentence in evidence[:2]:
            gaps.append({
                "source": f"Paper {paper_number}",
                "evidence": sentence,
                "interpretation": (
                    "This statement may indicate a limitation or an area "
                    "for further investigation. Validate it against the "
                    "full paper before treating it as a research gap."
                )
            })

    similarity = float(cosine_similarity(matrix[0], matrix[1])[0][0])

    return {
        "error": None,
        "similarity": round(similarity * 100, 1),
        "common_topics": common,
        "unique_topics_1": unique1,
        "unique_topics_2": unique2,
        "objectives": objectives,
        "methods": methods,
        "findings": findings,
        "limitations": limitations,
        "gaps": gaps
    }


# ============================================================
# COMPARE RESEARCH PAPERS
# ============================================================

@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare_papers():
    papers = []
    comparison = None
    analysis = None
    error = None

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT id, title
            FROM research_papers
            WHERE user_id = %s
            ORDER BY title
        """, (current_user.id,))

        papers = cursor.fetchall()

        if request.method == "POST":
            try:
                first_id = int(request.form.get("paper1", ""))
                second_id = int(request.form.get("paper2", ""))
            except (ValueError, TypeError):
                first_id = second_id = None

            if not first_id or not second_id:
                error = "Please select two valid research papers."
            elif first_id == second_id:
                error = "Please select two different papers."
            else:
                cursor.execute("""
                    SELECT id, title, filename, extracted_text
                    FROM research_papers
                    WHERE user_id = %s
                      AND id IN (%s, %s)
                """, (current_user.id, first_id, second_id))

                rows = {row[0]: row for row in cursor.fetchall()}

                if first_id not in rows or second_id not in rows:
                    error = "One or both papers could not be found."
                else:
                    comparison = [rows[first_id], rows[second_id]]
                    analysis = analyze_paper_comparison(
                        rows[first_id][3],
                        rows[second_id][3]
                    )

    except Exception:
        app.logger.exception("Paper comparison failed")
        error = "Unable to compare the papers. Please try again."
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

    return render_template(
        "compare.html",
        papers=papers,
        comparison=comparison,
        analysis=analysis,
        error=error
    )


# ============================================================
# VIEW INDIVIDUAL PAPER
# ============================================================

@app.route("/paper/<int:paper_id>")
@login_required
def view_paper(paper_id):
    paper = None
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                title,
                filename,
                extracted_text,
                uploaded_at
            FROM research_papers
            WHERE id = %s
              AND user_id = %s
            """,
            (paper_id, current_user.id)
        )

        paper = cursor.fetchone()

    except Exception as e:
        app.logger.exception(
            "Paper details database error for paper ID %s",
            paper_id
        )
        return (
            "Unable to load this research paper. Please try again.",
            500
        )

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    if paper is None:
        return "Research paper not found", 404

    return render_template("paper.html", paper=paper)


# ============================================================
# DOWNLOAD PAPER
# ============================================================

@app.route(
    "/download/<int:paper_id>"
)
@login_required
def download_paper(paper_id):

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(

            """
            SELECT filename
            FROM research_papers
            WHERE id = %s
            AND user_id = %s
            """,

            (
                paper_id,
                current_user.id
            )

        )

        paper = cursor.fetchone()

        cursor.close()

        connection.close()

        if not paper:

            return (
                "Research paper not found",
                404
            )

        filename = paper[0]

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )

        if not os.path.exists(filepath):

            return (
                "PDF file not found on server",
                404
            )

        return send_from_directory(

            app.config["UPLOAD_FOLDER"],

            filename,

            as_attachment=True

        )

    except Exception as e:

        print(
            "Download error:",
            e
        )

        return (
            "Failed to download paper",
            500
        )


# ============================================================
# UPDATE PAPER
# ============================================================

@app.route(
    "/update/<int:paper_id>",
    methods=["GET", "POST"]
)
@login_required
def update_paper(paper_id):

    # ========================================================
    # GET EXISTING PAPER
    # ONLY CURRENT USER'S PAPER
    # ========================================================

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(

            """
            SELECT
                id,
                title,
                filename,
                extracted_text,
                uploaded_at
            FROM research_papers
            WHERE id = %s
            AND user_id = %s
            """,

            (
                paper_id,
                current_user.id
            )

        )

        paper = cursor.fetchone()

        cursor.close()

        connection.close()

    except Exception as e:

        print(
            "Update paper database error:",
            e
        )

        return (
            "Database error",
            500
        )

    if not paper:

        return (
            "Research paper not found",
            404
        )

    # ========================================================
    # GET
    # ========================================================

    if request.method == "GET":

        return render_template(

            "update.html",

            paper=paper

        )

    # ========================================================
    # POST
    # ========================================================

    file = request.files.get("file")

    if not file or file.filename == "":

        return render_template(

            "update.html",

            paper=paper,

            error="Please select a PDF file."

        )

    if not file.filename.lower().endswith(".pdf"):

        return render_template(

            "update.html",

            paper=paper,

            error="Only PDF files are allowed."

        )

    # ========================================================
    # OLD FILE
    # ========================================================

    old_filename = paper[2]

    old_filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        old_filename

    )

    # ========================================================
    # NEW FILE
    # ========================================================

    original_filename = secure_filename(
        file.filename
    )

    new_filename = (

        str(current_user.id)
        + "_"
        + str(uuid.uuid4())
        + "_"
        + original_filename

    )

    new_filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        new_filename

    )

    file.save(new_filepath)

    # ========================================================
    # EXTRACT TEXT
    # ========================================================

    extracted_text = ""

    try:

        reader = PdfReader(
            new_filepath
        )

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                extracted_text += (
                    page_text + "\n"
                )

    except Exception as e:

        print(
            "Update PDF extraction error:",
            e
        )

        if os.path.exists(new_filepath):

            os.remove(new_filepath)

        return render_template(

            "update.html",

            paper=paper,

            error=(
                "Failed to extract "
                "text from PDF."
            )

        )

   
    # ========================================================
    # EMBEDDING
    # ========================================================

    embedding = generate_document_embedding(
        preprocess_pdf_text(extracted_text)
    )

    if embedding is None:
        if os.path.exists(new_filepath):
            os.remove(new_filepath)

        return render_template(
            "update.html",
            paper=paper,
            error="This PDF contains too little extractable text."
        )

   
    # ========================================================
    # UPDATE DATABASE
    # ========================================================

    new_title = os.path.splitext(original_filename)[0]

    # If a previously stored filename was uploaded again,
    # remove its existing user-ID and UUID prefix safely.
    parts = new_title.split("_", 2)

    if len(parts) == 3 and parts[0].isdigit():
        try:
            uuid.UUID(parts[1])
        except ValueError:
            pass
        else:
            new_title = parts[2]

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE research_papers
            SET
                title = %s,
                filename = %s,
                extracted_text = %s,
                embedding = %s,
                uploaded_at = CURRENT_TIMESTAMP
            WHERE id = %s
              AND user_id = %s
            """,
            (
                new_title,
                new_filename,
                preprocess_pdf_text(extracted_text),
                embedding,
                paper_id,
                current_user.id
            )
        )

        
        if cursor.rowcount != 1:
            connection.rollback()

            if os.path.exists(new_filepath):
                os.remove(new_filepath)

            return "Research paper not found", 404

        # Record the successful update in the audit log
        cursor.execute(
            """
            INSERT INTO research_audit_log
                (paper_id, user_id, action, details)
            VALUES (%s, %s, %s, %s)
            """,
            (
                paper_id,
                current_user.id,
                "UPDATE",
                "Research paper updated"
            )
        )

        # Save paper update and audit entry together
        connection.commit()


    except Exception:
        app.logger.exception("Update database error")

        if connection is not None:
            connection.rollback()

        if os.path.exists(new_filepath):
            os.remove(new_filepath)

        return "Failed to update research paper", 500

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    # ========================================================
    # DELETE OLD FILE AFTER SUCCESSFUL DATABASE UPDATE
    # ========================================================

    if old_filename != new_filename and os.path.exists(old_filepath):
        try:
            os.remove(old_filepath)
        except OSError:
            app.logger.exception("Could not remove the old PDF file")

    return redirect(url_for("repository"))


    
# ============================================================
# DELETE PAPER
# ============================================================

@app.route("/delete/<int:paper_id>", methods=["POST"])
@login_required
def delete_paper(paper_id):
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Find the paper belonging to the logged-in user
        cursor.execute(
            """
            SELECT title, filename
            FROM research_papers
            WHERE id = %s AND user_id = %s
            FOR UPDATE
            """,
            (paper_id, current_user.id)
        )

        paper = cursor.fetchone()

        if not paper:
            connection.rollback()
            return "Research paper not found", 404

        title, filename = paper

        # Record deletion before removing the paper
        cursor.execute(
            """
            INSERT INTO research_audit_log
                (paper_id, user_id, action, details)
            VALUES (%s, %s, %s, %s)
            """,
            (
                paper_id,
                current_user.id,
                "DELETE",
                f"Deleted research paper: {title}"
            )
        )

        # Delete only the current user's paper
        cursor.execute(
            """
            DELETE FROM research_papers
            WHERE id = %s AND user_id = %s
            """,
            (paper_id, current_user.id)
        )

        if cursor.rowcount != 1:
            connection.rollback()
            return "Research paper not found", 404

        # Save the audit entry and deletion together
        connection.commit()

        print("Paper deleted for:", current_user.username)

    except Exception:
        app.logger.exception("Delete paper error")

        if connection is not None:
            connection.rollback()

        return "Failed to delete research paper", 500

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    # Delete the physical PDF after the database transaction succeeds
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            app.logger.exception(
                "Paper deleted from database, but PDF removal failed"
            )

    return redirect(url_for("repository"))



@app.route("/ml-dashboard")
@login_required
def ml_dashboard():
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Statistics for the logged-in user only
        cursor.execute("""
            SELECT
                COUNT(*),
                COUNT(embedding),
                COUNT(*) FILTER (
                    WHERE embedding IS NULL
                ),
                COUNT(*) FILTER (
                    WHERE extracted_text IS NULL
                       OR BTRIM(extracted_text) = ''
                )
            FROM research_papers
            WHERE user_id = %s
        """, (current_user.id,))

        stats = cursor.fetchone()

        total_papers = stats[0]
        embedded_papers = stats[1]
        missing_embeddings = stats[2]
        missing_text = stats[3]

        # Only this user's recent uploads
        cursor.execute("""
            SELECT id, title, filename, uploaded_at
            FROM research_papers
            WHERE user_id = %s
            ORDER BY uploaded_at DESC NULLS LAST, id DESC
            LIMIT 5
        """, (current_user.id,))

        recent_papers = cursor.fetchall()

        return render_template(
            "ml_dashboard.html",
            total_papers=total_papers,
            embedded_papers=embedded_papers,
            missing_embeddings=missing_embeddings,
            missing_text=missing_text,
            recent_papers=recent_papers,
            error=None
        )

    except Exception:
        app.logger.exception(
            "Unable to load Research Intelligence Dashboard."
        )

        return render_template(
            "ml_dashboard.html",
            total_papers=0,
            embedded_papers=0,
            missing_embeddings=0,
            missing_text=0,
            recent_papers=[],
            error="Could not load your research statistics. Check the database connection."
        ), 500

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    
# ============================================================
# DATABASE ACTIVITY
# ============================================================

@app.route("/database-activity")
@login_required
def database_activity():
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                audit_id,
                paper_id,
                action,
                action_time,
                details
            FROM research_audit_log
            WHERE user_id = %s
            ORDER BY action_time DESC, audit_id DESC
            LIMIT 100
            """,
            (current_user.id,)
        )

        activities = cursor.fetchall()

        return render_template(
            "database_activity.html",
            activities=activities,
            error=None
        )

    except Exception:
        app.logger.exception(
            "Unable to load database activity"
        )

        return render_template(
            "database_activity.html",
            activities=[],
            error="Unable to load database activity. Please try again."
        ), 500

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ============================================================
# SMART RESEARCH WORKSPACE
# ============================================================

@app.route("/workspace", methods=["GET", "POST"])
@login_required
def workspace():
    connection = None
    cursor = None
    message = None
    error = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        if request.method == "POST":
            action = request.form.get("action", "").strip()

            if action == "create_collection":
                name = request.form.get(
                    "collection_name", ""
                ).strip()

                if not name:
                    error = "Enter a collection name."
                else:
                    cursor.execute(
                        """
                        INSERT INTO workspace_collections
                            (user_id, collection_name)
                        VALUES (%s, %s)
                        """,
                        (current_user.id, name)
                    )
                    connection.commit()
                    message = "Collection created successfully."

            elif action == "save_paper":
                paper_id = request.form.get("paper_id", type=int)
                collection_id = request.form.get(
                    "collection_id", type=int
                )
                notes = request.form.get("notes", "").strip()

                if not paper_id:
                    error = "Select a research paper."
                else:
                    # Confirm that the paper belongs to this user.
                    cursor.execute(
                        """
                        SELECT id
                        FROM research_papers
                        WHERE id = %s AND user_id = %s
                        """,
                        (paper_id, current_user.id)
                    )

                    if not cursor.fetchone():
                        error = "Paper not found in your repository."
                    elif collection_id:
                        cursor.execute(
                            """
                            SELECT collection_id
                            FROM workspace_collections
                            WHERE collection_id = %s AND user_id = %s
                            """,
                            (collection_id, current_user.id)
                        )

                        if not cursor.fetchone():
                            error = "Select a valid collection."

                    if not error:
                        cursor.execute(
                            """
                            INSERT INTO workspace_items
                                (user_id, paper_id, collection_id, notes)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (
                                current_user.id,
                                paper_id,
                                collection_id or None,
                                notes
                            )
                        )
                        connection.commit()
                        message = "Paper saved to your workspace."

            elif action == "update_note":
                item_id = request.form.get("item_id", type=int)
                notes = request.form.get("notes", "").strip()

                cursor.execute(
                    """
                    UPDATE workspace_items
                    SET notes = %s
                    WHERE item_id = %s AND user_id = %s
                    """,
                    (notes, item_id, current_user.id)
                )
                connection.commit()

                if cursor.rowcount:
                    message = "Notes updated successfully."
                else:
                    error = "Workspace item not found."

            elif action == "remove_paper":
                item_id = request.form.get("item_id", type=int)

                cursor.execute(
                    """
                    DELETE FROM workspace_items
                    WHERE item_id = %s AND user_id = %s
                    """,
                    (item_id, current_user.id)
                )
                connection.commit()

                if cursor.rowcount:
                    message = "Paper removed from workspace."
                else:
                    error = "Workspace item not found."

            else:
                error = "Unknown workspace action."

        # Load the logged-in user's research papers.
        cursor.execute(
            """
            SELECT id, title
            FROM research_papers
            WHERE user_id = %s
            ORDER BY id DESC
            """,
            (current_user.id,)
        )
        papers = cursor.fetchall()

        # Load this user's collections.
        cursor.execute(
            """
            SELECT collection_id, collection_name, created_at
            FROM workspace_collections
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (current_user.id,)
        )
        collections = cursor.fetchall()

        # Load saved papers and their notes.
        cursor.execute(
            """
            SELECT
                wi.item_id,
                wi.paper_id,
                rp.title,
                wc.collection_name,
                wi.notes,
                wi.saved_at
            FROM workspace_items wi
            JOIN research_papers rp
                ON rp.id = wi.paper_id
            LEFT JOIN workspace_collections wc
                ON wc.collection_id = wi.collection_id
               AND wc.user_id = wi.user_id
            WHERE wi.user_id = %s
              AND rp.user_id = %s
            ORDER BY wi.saved_at DESC
            """,
            (current_user.id, current_user.id)
        )
        saved_items = cursor.fetchall()

        return render_template(
            "workspace.html",
            papers=papers,
            collections=collections,
            saved_items=saved_items,
            message=message,
            error=error
        )

    except Exception:
        if connection:
            connection.rollback()

        app.logger.exception(
            "Unable to load Smart Research Workspace"
        )

        return (
            "Workspace could not be loaded. "
            "Check the Flask terminal for the error.",
            500
        )

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# ============================================================
# CITATION GENERATOR
# ============================================================

@app.route("/citations", methods=["GET", "POST"])
@login_required
def citations():
    details = None
    generated_citations = None
    error = None

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        authors = request.form.get("authors", "").strip()
        year = request.form.get("year", "").strip()
        journal = request.form.get("journal", "").strip()
        doi = request.form.get("doi", "").strip()

        details = {
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "doi": doi
        }

        if not title:
            error = "Paper title is required."
        elif year and (not year.isdigit() or not 1000 <= int(year) <= 2100):
            error = "Enter a valid publication year between 1000 and 2100."
        else:
            author_list = [
                name.strip()
                for name in authors.split(",")
                if name.strip()
            ]

            publication_year = year if year else "n.d."
            journal_text = f" {journal}." if journal else ""
            doi_text = f" {doi}" if doi else ""

            # APA-style reference
            if author_list:
                apa_authors = ", ".join(author_list)
            else:
                apa_authors = "Unknown author"

            apa = (
                f"{apa_authors} ({publication_year}). "
                f"{title}.{journal_text}{doi_text}"
            )

            # MLA-style reference
            mla_authors = (
                ", ".join(author_list)
                if author_list
                else "Unknown author"
            )

            mla = (
                f'{mla_authors}. "{title}."'
                f'{f" {journal}." if journal else ""}'
                f' {year + "." if year else "n.d."}'
                f'{doi_text}'
            )

            # IEEE-style reference
            ieee_authors = (
                ", ".join(author_list)
                if author_list
                else "Unknown author"
            )

            ieee = (
                f'{ieee_authors}, "{title},"'
                f'{f" {journal}," if journal else ""}'
                f' {year + "." if year else "n.d."}'
                f'{doi_text}'
            )

            generated_citations = {
                "APA": apa,
                "MLA": mla,
                "IEEE": ieee
            }

    return render_template(
        "citations.html",
        details=details,
        citations=generated_citations,
        error=error
    )



# ============================================================
# RESEARCH PAPER SUMMARIZER
# ============================================================

def extract_research_section(sentences, keywords, limit=3):
    """Find sentences that may describe a particular research section."""
    matches = []

    for sentence in sentences:
        lower_sentence = sentence.lower()

        if any(keyword in lower_sentence for keyword in keywords):
            if sentence not in matches:
                matches.append(sentence)

        if len(matches) >= limit:
            break

    return matches


@app.route("/summarizer", methods=["GET", "POST"])
@login_required
def summarizer():
    papers = []
    selected_paper = None
    summary = None
    key_points = []
    breakdown = {}
    error = None

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, title
            FROM research_papers
            WHERE user_id = %s
            ORDER BY uploaded_at DESC NULLS LAST
            """,
            (current_user.id,)
        )
        papers = cursor.fetchall()

        if request.method == "POST":
            paper_id = request.form.get("paper_id", type=int)

            if not paper_id:
                error = "Please select a research paper."
            else:
                cursor.execute(
                    """
                    SELECT id, title, extracted_text
                    FROM research_papers
                    WHERE id = %s AND user_id = %s
                    """,
                    (paper_id, current_user.id)
                )
                selected_paper = cursor.fetchone()

                if not selected_paper:
                    error = "Paper not found in your repository."

                elif not selected_paper[2] or not selected_paper[2].strip():
                    error = "This paper has no extractable text."

                else:
                    from sklearn.feature_extraction.text import TfidfVectorizer

                    text = selected_paper[2]

                    sentences = [
                        re.sub(r"\s+", " ", sentence).strip()
                        for sentence in re.split(r"(?<=[.!?])\s+", text)
                        if len(sentence.strip().split()) >= 6
                    ]

                    if not sentences:
                        error = "Not enough readable sentences to summarize."

                    else:
                        # Create an extractive summary and rank important sentences.
                        if len(sentences) <= 5:
                            selected_indices = list(range(len(sentences)))
                        else:
                            try:
                                vectorizer = TfidfVectorizer(
                                    stop_words="english",
                                    max_features=3000
                                )
                                matrix = vectorizer.fit_transform(sentences)
                                scores = matrix.sum(axis=1).A1

                                ranked = sorted(
                                    range(len(sentences)),
                                    key=lambda i: scores[i],
                                    reverse=True
                                )

                                chosen = set(ranked[:min(8, len(sentences))])
                                chosen.update(range(min(2, len(sentences))))
                                selected_indices = sorted(chosen)

                            except ValueError:
                                selected_indices = list(
                                    range(min(5, len(sentences)))
                                )

                        summary = " ".join(
                            sentences[i] for i in selected_indices
                        )

                        # Main key points: highest-ranked sentences.
                        key_points = [
                            sentences[i]
                            for i in selected_indices
                        ][:5]

                        # Identify likely sections using sentence clues.
                        breakdown = {
                            "Objectives": extract_research_section(
                                sentences,
                                [
                                    "objective", "aim of this study",
                                    "purpose of this study", "this paper aims",
                                    "we propose", "we investigate",
                                    "this study presents"
                                ]
                            ),
                            "Methodology": extract_research_section(
                                sentences,
                                [
                                    "methodology", "we use", "we used",
                                    "proposed method", "our approach",
                                    "dataset", "experiments were",
                                    "experimental setup", "we trained",
                                    "the model was trained"
                                ]
                            ),
                            "Results": extract_research_section(
                                sentences,
                                [
                                    "results show", "results indicate",
                                    "our results", "we found",
                                    "experimental results", "achieved",
                                    "outperformed", "accuracy",
                                    "performance improved"
                                ]
                            ),
                            "Conclusion": extract_research_section(
                                sentences,
                                [
                                    "in conclusion", "we conclude",
                                    "this paper concludes", "in summary",
                                    "future work", "these findings",
                                    "the findings suggest"
                                ]
                            )
                        }

    except Exception:
        app.logger.exception("Research paper summarization failed")
        error = "Unable to summarize the paper. Please try again."

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    return render_template(
        "summarizer.html",
        papers=papers,
        selected_paper=selected_paper,
        summary=summary,
        key_points=key_points,
        breakdown=breakdown,
        error=error
    )


# ============================================================
# DOWNLOAD RESEARCH SUMMARY
# ============================================================

@app.route("/summarizer/download", methods=["POST"])
@login_required
def download_summary():
    from io import BytesIO
    from flask import send_file

    title = request.form.get("title", "Research Summary").strip()
    summary = request.form.get("summary", "").strip()
    key_points = request.form.getlist("key_points")
    objectives = request.form.getlist("objectives")
    methodology = request.form.getlist("methodology")
    results = request.form.getlist("results")
    conclusion = request.form.getlist("conclusion")

    if not summary:
        return "No summary available to download.", 400

    # Confirm that the paper belongs to the logged-in user.
    paper_id = request.form.get("paper_id", type=int)

    if not paper_id:
        return "Invalid paper selection.", 400

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM research_papers
            WHERE id = %s AND user_id = %s
            """,
            (paper_id, current_user.id)
        )

        if not cursor.fetchone():
            return "Research paper not found.", 404

    except Exception:
        app.logger.exception("Summary download validation failed")
        return "Unable to validate the paper.", 500

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    sections = [
        "RESEARCH PAPER SUMMARY",
        title,
        "",
        "SUMMARY",
        summary,
        "",
        "KEY POINTS",
        *[f"- {point}" for point in key_points],
        "",
        "OBJECTIVES",
        *[f"- {item}" for item in objectives],
        "",
        "METHODOLOGY",
        *[f"- {item}" for item in methodology],
        "",
        "RESULTS",
        *[f"- {item}" for item in results],
        "",
        "CONCLUSION",
        *[f"- {item}" for item in conclusion],
    ]

    content = "\n".join(sections)
    filename = "research_summary.txt"

    return send_file(
        BytesIO(content.encode("utf-8")),
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name=filename
    )





# ============================================================
# AI RESEARCH CHATBOX
# ============================================================

@app.route("/research-chat", methods=["GET", "POST"])
@login_required
def research_chat():
    papers = []
    answer = None
    question = ""
    error = None
    selected_paper_id = None

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, title
            FROM research_papers
            WHERE user_id = %s
            ORDER BY uploaded_at DESC NULLS LAST
            """,
            (current_user.id,)
        )
        papers = cursor.fetchall()

        if request.method == "POST":
            question = request.form.get("question", "").strip()
            selected_paper_id = request.form.get("paper_id", type=int)

            if not question:
                error = "Please enter a question."
            elif len(question) > 2000:
                error = "Please keep your question under 2,000 characters."
            elif not selected_paper_id:
                error = "Please select a research paper."
            elif not os.environ.get("OPENAI_API_KEY"):
                error = "API key is not configured in this terminal."
            else:
                cursor.execute(
                    """
                    SELECT title, extracted_text
                    FROM research_papers
                    WHERE id = %s AND user_id = %s
                    """,
                    (selected_paper_id, current_user.id)
                )
                paper = cursor.fetchone()

                if not paper:
                    error = "Selected paper was not found."
                elif not paper[1] or not paper[1].strip():
                    error = "The selected paper has no extractable text."
                else:
                    from openai import OpenAI

                    client = OpenAI()

                    response = client.responses.create(
                        model="gpt-4o-mini",
                        instructions=(
                            "Answer questions about the provided research "
                            "paper. Do not invent facts or citations. If the "
                            "paper does not contain the answer, say so."
                        ),
                        input=(
                            "Paper title: " + paper[0] +
                            "\n\nPaper text:\n" + paper[1][:18000] +
                            "\n\nQuestion: " + question
                        )
                    )
                    answer = response.output_text

    except Exception:
        app.logger.exception("AI Research Chat failed")
        error = (
            "The chat request failed. Check the Flask terminal for details."
        )

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    return render_template(
        "research_chat.html",
        papers=papers,
        answer=answer,
        question=question,
        error=error,
        selected_paper_id=selected_paper_id
    )


# ============================================================
# STEP 6: RESEARCH GAP FINDER
# ============================================================

@app.route("/research-gaps", methods=["GET", "POST"])
@login_required
def research_gaps():
    papers = []
    selected_paper = None
    gap_candidates = []
    related_papers = []
    error = None

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Load papers belonging to the logged-in user.
        cursor.execute(
            """
            SELECT id, title
            FROM research_papers
            WHERE user_id = %s
            ORDER BY uploaded_at DESC NULLS LAST
            """,
            (current_user.id,)
        )
        papers = cursor.fetchall()

        if request.method == "POST":
            paper_id = request.form.get("paper_id", type=int)

            if not paper_id:
                error = "Please select a research paper."
            else:
                cursor.execute(
                    """
                    SELECT id, title, extracted_text
                    FROM research_papers
                    WHERE id = %s AND user_id = %s
                    """,
                    (paper_id, current_user.id)
                )
                selected_paper = cursor.fetchone()

                if not selected_paper:
                    error = "Selected paper was not found."

                elif not selected_paper[2] or not selected_paper[2].strip():
                    error = "The selected paper has no extractable text."

                else:
                    text = selected_paper[2]

                    # Detect statements about limitations and future work.
                    gap_phrases = [
                        "future work",
                        "future research",
                        "further research",
                        "further investigation",
                        "limitations",
                        "limitation",
                        "remains unclear",
                        "has not been explored",
                        "has not been studied",
                        "requires further study",
                        "future studies",
                        "not addressed",
                        "open problem"
                    ]

                    sentences = [
                        re.sub(r"\s+", " ", sentence).strip()
                        for sentence in re.split(
                            r"(?<=[.!?])\s+", text
                        )
                        if len(sentence.strip().split()) >= 6
                    ]

                    for sentence in sentences:
                        if any(
                            phrase in sentence.lower()
                            for phrase in gap_phrases
                        ):
                            gap_candidates.append(sentence)

                    # Find other papers from the same user's repository.
                    cursor.execute(
                        """
                        SELECT id, title, extracted_text
                        FROM research_papers
                        WHERE user_id = %s
                          AND id <> %s
                          AND extracted_text IS NOT NULL
                          AND LENGTH(TRIM(extracted_text)) > 0
                        ORDER BY uploaded_at DESC NULLS LAST
                        """,
                        (current_user.id, paper_id)
                    )
                    other_papers = cursor.fetchall()

                    related_papers = [
                        {"id": row[0], "title": row[1]}
                        for row in other_papers
                    ]

                    # Compare topics using TF-IDF if other papers exist.
                    if other_papers:
                        from sklearn.feature_extraction.text import (
                            TfidfVectorizer
                        )

                        documents = [text[:50000]] + [
                            row[2][:50000] for row in other_papers
                        ]

                        try:
                            vectorizer = TfidfVectorizer(
                                stop_words="english",
                                max_features=2500,
                                ngram_range=(1, 2)
                            )

                            matrix = vectorizer.fit_transform(documents)
                            terms = vectorizer.get_feature_names_out()

                            selected_scores = matrix[0].toarray().ravel()
                            other_scores = matrix[1:].toarray().mean(axis=0)

                            differences = other_scores - selected_scores
                            ranked_indices = differences.argsort()[::-1]

                            added = 0

                            for index in ranked_indices:
                                term = terms[index]

                                if (
                                    differences[index] <= 0
                                    or len(term) < 4
                                    or term in text.lower()
                                ):
                                    continue

                                gap_candidates.append(
                                    "Possible topic to investigate: '"
                                    + term
                                    + "'. This topic has a higher TF-IDF "
                                    "weight in the other uploaded papers "
                                    "than in the selected paper. Review the "
                                    "papers to determine whether it represents "
                                    "a genuine research gap."
                                )

                                added += 1

                                if added >= 8:
                                    break

                        except ValueError:
                            app.logger.warning(
                                "TF-IDF comparison could not be performed."
                            )

                    # Remove duplicate statements.
                    gap_candidates = list(dict.fromkeys(gap_candidates))

                    if not gap_candidates:
                        gap_candidates.append(
                            "No clear candidate gaps were detected "
                            "automatically. Review the paper's limitations, "
                            "discussion, and future-work sections manually."
                        )

    except Exception:
        app.logger.exception("Research Gap Finder failed")
        error = "Unable to analyze research gaps. Check the Flask terminal."

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "research_gaps.html",
        papers=papers,
        selected_paper=selected_paper,
        gap_candidates=gap_candidates,
        related_papers=related_papers,
        error=error
    )



# ============================================================
# NORMAL DOUBT-CLEARING CHATBOT
# ============================================================

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)

