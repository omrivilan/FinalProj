import database_mongo as database
import auth_manager
import utils


def register_user(username, email, password):
    """Registers a new user after validating input and checking for duplicates.

    This function verifies required fields, checks email format and password length,
    ensures that the username and email are not already taken, hashes the password,
    and creates a new user record in the database.

    Args:
        username (str): Desired username for the new account.
        email (str): Email address to associate with the account.
        password (str): Plaintext password for the account.

    Returns:
        dict: A response dictionary containing:
            - 'success' (bool): True if registration succeeded.
            - 'message' (str): Explanation of the outcome.
    """

    print(f"Attempting to register user: {username} with email: {email}")
    
    # Validate inputs
    if not username or not email or not password:
        print("Registration failed: Missing required fields")
        return {"success": False, "message": "All fields are required"}
    
    if not utils.is_valid_email(email):
        print("Registration failed: Invalid email format")
        return {"success": False, "message": "Invalid email format"}
    
    if len(password) < 8:
        print("Registration failed: Password too short")
        return {"success": False, "message": "Password must be at least 8 characters long"}
    
    # Check if username is already taken
    existing_user = database.get_user_by_username(username)
    if existing_user:
        print(f"Registration failed: Username {username} already exists")
        return {"success": False, "message": "Username already exists"}
    
    # Check if email is already registered
    existing_email = database.get_user_by_email(email)
    if existing_email:
        print(f"Registration failed: Email {email} already registered")
        return {"success": False, "message": "Email is already registered"}
    
    # Hash the password
    password_hash = auth_manager.hash_password(password)
    
    # Create the user
    print("Attempting to create user in database...")
    result = database.create_user(username, email, password_hash)
    print(f"Database creation result: {result}")
    return result

def request_password_reset(email):
    """Request a password reset for a user.

    If the provided email is associated with a user account, this function generates a
    secure reset token, stores it in the database with a 1-hour expiration, and initiates
    the password reset process (e.g., via email). For security, it returns a generic
    success message regardless of whether the email is registered.

    Args:
        email (str): The user's registered email address.

    Returns:
        dict: A response indicating the reset request was accepted.
            - 'success' (bool): Always True.
            - 'message' (str): Generic success message.
    """

    # Check if email exists
    user = database.get_user_by_email(email)
    if not user:
        # For security reasons, don't reveal if email exists or not
        return {"success": True, "message": "If your email is registered, you will receive reset instructions"}
    
    # Generate a reset token
    reset_token = auth_manager.generate_password_reset_token()
    
    # Store the token
    database.store_reset_token(email, reset_token)
    
    # In a real application, send an email with the reset token
    # Here we'll just return it for testing purposes
    # utils.send_reset_email(email, reset_token)
    
    # For demo purposes, show the reset token in the console
    # In production, this would be sent via email and not displayed
    print(f"Password reset token for {email}: {reset_token}")
    
    return {"success": True, "message": "Reset instructions sent to your email"}

def reset_password(reset_token, new_password):
    """Reset a user's password using a valid reset token.

    This function verifies the validity of the reset token, hashes the new password,
    updates the user's password in the database, and then invalidates the token
    to prevent reuse.

    Args:
        reset_token (str): The unique token provided for password reset.
        new_password (str): The new password to be set.

    Returns:
        dict: A response indicating the success or failure of the password reset.
            - 'success' (bool): True if the password was successfully reset.
            - 'message' (str): A description of the result.
    """

    # Validate the token
    token_validation = database.validate_reset_token(reset_token)
    
    if not token_validation["success"]:
        return token_validation
    
    # Get the user by email
    email = token_validation["email"]
    user = database.get_user_by_email(email)
    
    if not user:
        return {"success": False, "message": "User not found"}
    
    # Hash the new password
    new_password_hash = auth_manager.hash_password(new_password)
    
    # Update the user's password
    result = database.update_user_password(user["username"], new_password_hash)
    
    # Invalidate the reset token
    database.invalidate_reset_token(reset_token)
    
    return result

def social_login(email, provider):
    """Handles login or registration using a third-party social provider.

    If the user exists, the method verifies that the login provider matches
    the registered one. If not, it returns an error. If the user does not exist,
    a new account is created with a unique username and a random password.

    Args:
        email (str): The email address associated with the social login.
        provider (str): The name of the social provider (e.g., "google").

    Returns:
        dict: A response dictionary containing:
            - 'success' (bool): True if login or registration succeeded.
            - 'message' (str): Explanation of the result.
            - 'username' (str, optional): Assigned username if login/registration was successful.
    """

    # Check if user exists
    user = database.get_user_by_email(email)
    
    if user:
        # User exists, check if they registered with this provider
        if user["provider"] == provider or user["provider"] == "local":
            # Allow login
            return {"success": True, "message": "Login successful", "username": user["username"]}
        else:
            # User registered with a different provider
            return {"success": False, "message": f"This email is already registered using {user['provider']}"}
    else:
        # User doesn't exist, create a new account
        # Generate a username from email
        base_username = email.split('@')[0]
        username = base_username
        
        # Check if username exists, append numbers if needed
        count = 1
        while database.get_user_by_username(username):
            username = f"{base_username}{count}"
            count += 1
        
        # Create random password for the account (it won't be used for login)
        random_password = utils.generate_random_password()
        password_hash = auth_manager.hash_password(random_password)
        
        # Create the user
        result = database.create_user(username, email, password_hash, provider=provider)
        
        if result["success"]:
            return {"success": True, "message": "Account created successfully", "username": username}
        else:
            return result
