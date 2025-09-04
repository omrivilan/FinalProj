import re
import random
import string
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def is_valid_email(email):
    """
    Validate email format.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email format is valid, False otherwise.
    """
    # Basic email format validation using regex
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(email_pattern, email))

def generate_random_password(length=12):
    """Generate a secure random password.

    This function creates a password containing a mix of uppercase letters,
    lowercase letters, digits, and special characters. It ensures at least
    one character from each category is included.

    Args:
        length (int, optional): The total length of the generated password.
            Must be at least 4 to accommodate one character from each category.
            Defaults to 12.

    Returns:
        str: A securely generated random password.
    """
    # Include uppercase, lowercase, digits, and special characters
    characters = string.ascii_letters + string.digits + string.punctuation
    # Ensure at least one character from each category
    password = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
        random.choice(string.punctuation)
    ]
    # Fill the rest of the password length with random characters
    password.extend(random.choice(characters) for _ in range(length - 4))
    # Shuffle the password characters
    random.shuffle(password)
    return ''.join(password)

def send_reset_email(email, reset_token):
    """Send a password reset email to a user.

    Constructs and sends an email containing a password reset code
    using SMTP settings from environment variables. Supports both
    plain-text and HTML formats.

    Args:
        email (str): The recipient's email address.
        reset_token (str): The reset token to include in the email.

    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """

    # Email settings from environment variables
    smtp_server = os.getenv("SMTP_SERVER", "smtp.example.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    from_email = os.getenv("FROM_EMAIL", "noreply@example.com")
    app_name = os.getenv("APP_NAME", "Stock Market AI Chatbot")
    
    # Create email message
    message = MIMEMultipart("alternative")
    message["Subject"] = f"{app_name} - Password Reset"
    message["From"] = from_email
    message["To"] = email
    
    # Create the plain-text and HTML version of your message
    text = f"""
    Hello,
    
    You have requested to reset your password for {app_name}.
    
    Your reset code is: {reset_token}
    
    If you did not request a password reset, please ignore this email.
    
    Best regards,
    {app_name} Team
    """
    
    html = f"""
    <html>
      <body>
        <p>Hello,</p>
        <p>You have requested to reset your password for <strong>{app_name}</strong>.</p>
        <p>Your reset code is: <strong>{reset_token}</strong></p>
        <p>If you did not request a password reset, please ignore this email.</p>
        <p>Best regards,<br>{app_name} Team</p>
      </body>
    </html>
    """
    
    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    
    # Add HTML/plain-text parts to MIMEMultipart message
    message.attach(part1)
    message.attach(part2)
    
    try:
        # Create secure connection with server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, email, message.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False
