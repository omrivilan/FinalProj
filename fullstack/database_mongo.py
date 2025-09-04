from datetime import datetime
import time
from bson.objectid import ObjectId
from db import users_col, reset_tokens_col, users_chat_col

def get_user_by_username(username):
    """Retrieves a user document from the database by username.

    Args:
        username (str): The username to search for.

    Returns:
        dict or None: The user document if found; otherwise, None.
    """

    return users_col.find_one({"username": username})

def get_user_by_email(email):
    """Retrieves a user document from the database by email address.

    Args:
        email (str): The email address to search for.

    Returns:
        dict or None: The user document if found; otherwise, None.
    """

    return users_col.find_one({"email": email})

def create_user(username, email, password_hash, provider="local"):
    """Creates a new user account in the database after validating uniqueness.

    This function checks whether the username or email already exists. If both are unique,
    it inserts the new user into the `users_col` collection with metadata like provider and creation time.

    Args:
        username (str): The desired username for the new user.
        email (str): The user's email address.
        password_hash (str): The securely hashed password.
        provider (str, optional): The authentication provider ("local", "google", etc.). Defaults to "local".

    Returns:
        dict: A response dictionary with 'success' (bool) and 'message' (str) keys describing the result.
    """

    print(f"Checking if username {username} exists...")
    if users_col.find_one({"username": username}):
        print(f"Username {username} already exists")
        return {"success": False, "message": "Username already exists"}

    print(f"Checking if email {email} exists...")
    if users_col.find_one({"email": email}):
        print(f"Email {email} already registered")
        return {"success": False, "message": "Email already registered"}

    user_data = {
        "username": username,
        "email": email,
        "password": password_hash,
        "provider": provider,
        "created_at": datetime.utcnow().isoformat()
    }

    print("Attempting to insert user into database...")
    try:
        result = users_col.insert_one(user_data)
        print(f"Insert successful, document ID: {result.inserted_id}")
        return {"success": True, "message": "User created successfully"}
    except Exception as e:
        print(f"Error inserting user: {str(e)}")
        return {"success": False, "message": f"Database error: {str(e)}"}

def update_user_password(username, new_password_hash):
    """Updates the password hash for a user identified by username.

    Args:
        username (str): The username of the user whose password is being updated.
        new_password_hash (str): The new hashed password to store.

    Returns:
        dict: A response dictionary with 'success' (bool) and 'message' (str) indicating the outcome.
    """

    result = users_col.update_one(
        {"username": username},
        {"$set": {"password": new_password_hash}}
    )

    if result.matched_count == 0:
        return {"success": False, "message": "User not found"}
    
    return {"success": True, "message": "Password updated successfully"}

def store_reset_token(email, token):
    """Stores or updates a reset token for a given email with a 1-hour expiration.

    This function upserts the token and expiration timestamp into the `reset_tokens_col` collection.

    Args:
        email (str): The email address associated with the reset request.
        token (str): The reset token to store.

    Returns:
        dict: A response dictionary with 'success': True.
    """

    expiration = time.time() + 3600  # 1 hour from now
    reset_tokens_col.update_one(
        {"email": email},
        {"$set": {"token": token, "expiration": expiration}},
        upsert=True
    )
    return {"success": True}

def validate_reset_token(token):
    """Validates a reset token and checks its expiration.

    This function ensures the token exists and is still valid. If expired,
    it deletes the token from the database.

    Args:
        token (str): The reset token to validate.

    Returns:
        dict: A response dictionary containing:
            - 'success' (bool): Whether the token is valid.
            - 'message' (str, optional): Reason for failure if invalid.
            - 'email' (str, optional): Associated email if valid.
    """

    token_data = reset_tokens_col.find_one({"token": token})
    if not token_data:
        return {"success": False, "message": "Invalid reset code"}

    if token_data["expiration"] < time.time():
        reset_tokens_col.delete_one({"_id": token_data["_id"]})
        return {"success": False, "message": "Reset code has expired"}

    return {"success": True, "email": token_data["email"]}

def invalidate_reset_token(token):
    """Deletes a reset token from the database to prevent reuse.

    Args:
        token (str): The reset token to invalidate.

    Returns:
        dict: A response dictionary with 'success': True.
    """

    reset_tokens_col.delete_one({"token": token})
    return {"success": True}
def create_chat(username, chat_history, ts=None):
    """Creates a new chat session for the given user and stores it in the database.

    Args:
        username (str): The username associated with the chat session.
        chat_history (list): A list of message dictionaries representing the conversation.
        ts (datetime, optional): Timestamp for when the chat was created. Defaults to current UTC time.

    Returns:
        ObjectId: The MongoDB document ID of the newly inserted chat session.
    """

    doc = {
        "username": username,
        "chat_history": chat_history,
        "last_updated": ts or datetime.utcnow()
    }
    result = users_chat_col.insert_one(doc)
    return result.inserted_id


def get_chats_by_user(username):
    """Fetches all chat sessions' metadata for a user, excluding chat history.

    This function retrieves all chat documents for the given user, returning only
    the `_id` and `last_updated` fields, sorted by most recent activity.

    Args:
        username (str): The username whose chat sessions should be retrieved.

    Returns:
        list: A list of chat metadata documents (without chat_history).
    """

    cursor = users_chat_col.find(
        {"username": username},
        {"chat_history": 0}
    ).sort("last_updated", -1)
    return list(cursor)


def update_chat(chat_id, chat_history, ts=None):
    """Updates an existing chat session's history and timestamp by its document ID.

    This function overwrites the `chat_history` and updates the `last_updated` field.
    If no document matches the given ID, nothing is modified.

    Args:
        chat_id (str): The MongoDB `_id` of the chat session to update.
        chat_history (list): The new list of chat messages to store.
        ts (datetime, optional): Optional timestamp for update. Defaults to current UTC time.

    Returns:
        int: The number of documents modified (0 if not found, 1 if updated).
    """

    result = users_chat_col.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {
            "chat_history": chat_history,
            "last_updated": ts or datetime.utcnow()
        }}
    )
    return result.modified_count
def get_chat_by_id(chat_id):
    """Retrieves a single chat session document, including its full history, by document ID.

    Args:
        chat_id (str): The MongoDB `_id` of the chat session to retrieve.

    Returns:
        dict or None: The chat session document if found; otherwise, None.
    """
    
    return users_chat_col.find_one({"_id": ObjectId(chat_id)})

