import bcrypt
import os
import jwt
from datetime import datetime, timedelta
import uuid
import database_mongo

# Secret key for JWT tokens
JWT_SECRET = os.getenv("JWT_SECRET", "default_secret_key_change_in_production")

def hash_password(password):
    """Hashes a plaintext password using bcrypt.

    Args:
        password (str): The plaintext password to hash.

    Returns:
        str: The hashed password as a UTF-8 decoded string.
    """

    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    """Verifies a plaintext password against its bcrypt hash.

    Args:
        password (str): The plaintext password to verify.
        hashed_password (str): The bcrypt-hashed password to compare against.

    Returns:
        bool: True if the password matches the hash, False otherwise.
    """

    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def authenticate_user(username, password):
    """Authenticates a user using a provided username and password.

    This function checks whether the user exists and whether the password
    matches the stored bcrypt hash. It returns a success flag and an appropriate message.

    Args:
        username (str): The username to authenticate.
        password (str): The plaintext password to verify.

    Returns:
        dict: A response dictionary containing:
            - 'success' (bool): True if authentication succeeds.
            - 'message' (str): Description of the outcome.
            - 'user' (dict, optional): User document if authentication is successful.
    """

    user = database_mongo.get_user_by_username(username)
    
    if not user:
        return {"success": False, "message": "Invalid username or password"}
    
    if not verify_password(password, user["password"]):
        return {"success": False, "message": "Invalid username or password"}
    
    return {"success": True, "message": "Authentication successful", "user": user}

def generate_auth_token(username):
    """Generates a signed JWT token for an authenticated user.

    The token includes subject (`sub`), issued-at time (`iat`), expiration (`exp`),
    and a unique token identifier (`jti`), valid for 30 days.

    Args:
        username (str): The username to embed as the subject in the token.

    Returns:
        str: A JWT token string signed with the configured secret key.
    """

    expiration = datetime.utcnow() + timedelta(days=30)
    payload = {
        "sub": username,
        "exp": expiration,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token

def validate_auth_token(token):
    """Validates a JWT token and returns the associated user if valid.

    This function decodes the token using the configured secret key and retrieves
    the corresponding user from the database. It handles token expiration and
    invalid signature errors gracefully.

    Args:
        token (str): The JWT token to validate.

    Returns:
        dict or None: The user document if the token is valid; otherwise, None.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        username = payload["sub"]
        user = database_mongo.get_user_by_username(username)
        return user
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception:
        return None

def generate_password_reset_token():
    """Generates a random UUID token for password reset purposes.

    Returns:
        str: A randomly generated UUID4 string.
    """
    return str(uuid.uuid4())
