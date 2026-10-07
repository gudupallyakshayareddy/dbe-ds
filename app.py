from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    send_from_directory
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

print("Embedding model loaded!")


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
    # EMBEDDING
    # ========================================================

    embedding = None

    if extracted_text.strip():

        embedding = embedding_model.encode(

            extracted_text

        )

        embedding = np.asarray(

            embedding,

            dtype=np.float32

        )

        print(
            "Embedding generated successfully!"
        )

    else:

        print(
            "No text found in PDF."
        )

    # ========================================================
    # DATABASE STORAGE
    # ========================================================

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        title = os.path.splitext(
            original_filename
        )[0]

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
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,

            (
                current_user.id,
                title,
                unique_filename,
                extracted_text,
                embedding
            )

        )

        connection.commit()

        cursor.close()

        connection.close()

        print(
            "Paper stored for user:",
            current_user.username
        )

    except Exception as e:

        print(
            "Database error:",
            e
        )

        if os.path.exists(filepath):

            os.remove(filepath)

        return render_template(

            "index.html",

            filename="",

            extracted_text=""

        )

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
                query
            )

            query_embedding = np.asarray(
                query_embedding,
                dtype=np.float32
            )

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

                login_user(
                    logged_in_user
                )

                print(
                    "Login successful:",
                    username
                )

                return redirect(

                    url_for(
                        "upload_file"
                    )

                )

            return render_template(

                "login.html",

                error=(
                    "Invalid username "
                    "or password."
                )

            )

        except Exception as e:

            print(
                "Login error:",
                e
            )

            return render_template(

                "login.html",

                error=(
                    "Login failed. "
                    "Please try again."
                )

            )

    return render_template(
        "login.html"
    )


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

        cursor.close()

        connection.close()

        print(
            "Repository loaded for:",
            current_user.username
        )

        print(
            "Papers found:",
            len(papers)
        )

    except Exception as e:

        print(
            "Repository database error:",
            e
        )

    return render_template(

        "repository.html",

        papers=papers

    )


# ============================================================
# VIEW INDIVIDUAL PAPER
# ============================================================

@app.route(
    "/paper/<int:paper_id>"
)
@login_required
def view_paper(paper_id):

    paper = None

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
            "Paper details database error:",
            e
        )

    if not paper:

        return (
            "Research paper not found",
            404
        )

    return render_template(

        "paper.html",

        paper=paper

    )


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

    embedding = None

    if extracted_text.strip():

        embedding = embedding_model.encode(

            extracted_text

        )

        embedding = np.asarray(

            embedding,

            dtype=np.float32

        )

    # ========================================================
    # UPDATE DATABASE
    # ========================================================

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        new_title = os.path.splitext(
            original_filename
        )[0]

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
                extracted_text,
                embedding,
                paper_id,
                current_user.id
            )

        )

        connection.commit()

        cursor.close()

        connection.close()

    except Exception as e:

        print(
            "Update database error:",
            e
        )

        if os.path.exists(new_filepath):

            os.remove(new_filepath)

        return (
            "Failed to update research paper",
            500
        )

    # ========================================================
    # DELETE OLD FILE
    # ========================================================

    if (

        old_filename != new_filename

        and

        os.path.exists(old_filepath)

    ):

        os.remove(old_filepath)

    return redirect(

        url_for(
            "repository"
        )

    )


# ============================================================
# DELETE PAPER
# ============================================================

@app.route(
    "/delete/<int:paper_id>",
    methods=["POST"]
)
@login_required
def delete_paper(paper_id):

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        # ====================================================
        # GET FILE
        # ONLY CURRENT USER'S PAPER
        # ====================================================

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

        if not paper:

            cursor.close()

            connection.close()

            return (
                "Research paper not found",
                404
            )

        filename = paper[0]

        # ====================================================
        # DELETE DATABASE RECORD
        # ====================================================

        cursor.execute(

            """
            DELETE FROM research_papers
            WHERE id = %s
            AND user_id = %s
            """,

            (
                paper_id,
                current_user.id
            )

        )

        connection.commit()

        cursor.close()

        connection.close()

        # ====================================================
        # DELETE PDF
        # ====================================================

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )

        if os.path.exists(filepath):

            os.remove(filepath)

        print(
            "Paper deleted for:",
            current_user.username
        )

        return redirect(

            url_for(
                "repository"
            )

        )

    except Exception as e:

        print(
            "Delete paper error:",
            e
        )

        return (
            "Failed to delete research paper",
            500
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=False
    )