import streamlit as st
import os
import extra_streamlit_components as stx
from datetime import datetime, timedelta
import auth_manager
import user_management
import utils
# fullstack/app.py
from database_mongo import create_chat, get_chats_by_user
import os, sys

# ① Compute the absolute path to the project root (parent of this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# ② Inject it into Python's import search path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MainStreamlit_forecast import AppManager



# Set page configuration
st.set_page_config(
    page_title="Stock Market AI Chatbot - Login",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables if they don't exist
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "reset_password" not in st.session_state:
    st.session_state.reset_password = False
if "registration" not in st.session_state:
    st.session_state.registration = False
if "login_error" not in st.session_state:
    st.session_state.login_error = None

# Initialize cookie manager
cookie_manager = stx.CookieManager()

# Check if user is already authenticated via cookie
if "cookie_checked" not in st.session_state:
    st.session_state.cookie_checked = False

if not st.session_state.authenticated and not st.session_state.cookie_checked:
    auth_cookie = cookie_manager.get("auth_token")
    st.session_state.cookie_checked = True

    if auth_cookie:
        user = auth_manager.validate_auth_token(auth_cookie)
        if user:
            st.session_state.authenticated = True
            st.session_state.username = user["username"]




# Main app logic
def main():
    """Main entry point for the Streamlit app.

    Routes the user to the appropriate interface based on session state:
    login, registration, password reset, or main chatbot interface.

    Returns:
        None
    """

    if st.session_state.authenticated:
        display_main_app()
    elif st.session_state.reset_password:
        display_password_reset()
    elif st.session_state.registration:
        display_registration()
    else:
        display_login()

def display_login():
    """Display the login form and handle user authentication.

    This function renders a styled login form using Streamlit. It captures the user's
    username and password, verifies credentials via the authentication manager, and
    establishes session state variables upon successful login. Additionally, it manages
    a 'remember me' option by setting an authentication token in cookies, and provides
    a signup button to switch to the registration view.

    Returns:
        None
    """

    # Custom CSS for styling
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF0000;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }

    .login-box {
        background-color: #fff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 2px solid #e5e5e5;
    }

    .continue-btn button,
    .center-signup button {
        background-color: white !important;
        color: black !important;
        padding: 0.8rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        border: 2px solid #333 !important;
        width: 100% !important;
    }

    .signup-label {
        text-align: center;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .center-signup {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
    }

    .center-signup button {
        width: 200px !important;  /* or any fixed size */
        text-align: center !important;
    }  
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="main-title">Hello There! Let\'s get to know each other ☺️</div>', unsafe_allow_html=True)

    # Centered login box using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            # st.markdown('<div class="login-box">', unsafe_allow_html=True)

            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                remember_me = st.checkbox("Remember me", value=True)

                st.markdown('<div class="continue-btn">', unsafe_allow_html=True)
                submit = st.form_submit_button("Continue")
                st.markdown('</div>', unsafe_allow_html=True)

                if submit:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        login_result = auth_manager.authenticate_user(username, password)
                        if login_result["success"]:
                            st.session_state.authenticated = True
                            st.session_state.username = username

                            new_id = create_chat(username, [], None)
                            st.session_state.current_chat_id = str(new_id)
                            st.session_state.chat_history = []
                            st.session_state.available_chats = sorted(
                                get_chats_by_user(username),
                                key=lambda m: m["last_updated"],
                                reverse=True
                            )
                            st.session_state.selected_chat_idx = 0

                            if remember_me:
                                expiry = datetime.now() + timedelta(minutes=3)
                                token = auth_manager.generate_auth_token(username)
                                cookie_manager.set("auth_token", token, expires_at=expiry)

                            st.success("Login successful")
                            st.rerun()
                        else:
                            st.error(login_result["message"])

            # st.markdown('</div>', unsafe_allow_html=True)

    # Sign up section
    # Center "Don't have an account?" and Sign Up button together
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1,1,1,1, 1, 1,1,1,1])
    st.markdown('<div class="signup-label">Don\'t have an account?</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="center-signup">', unsafe_allow_html=True)
        if st.button("Sign Up", key="signup_trigger", use_container_width=True):
            st.session_state.registration = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def display_registration():
    """Display the user registration form and handle account creation.

    This function renders a styled registration form using Streamlit. It collects user
    information such as username, email, and password, performs basic validation, and 
    registers the user via the user management system. Upon successful registration, 
    it automatically logs the user in and initializes a new chat session.

    Returns:
        None
    """

    # Reuse the same CSS from login page
    st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #333 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        padding: 0.8rem !important;
        font-size: 1rem !important;
        border-radius: 5px !important;
        border: 1px solid #ccc !important;
    }
    
    /* Continue button styling */
    .continue-btn button {
        background-color: #10B981 !important;
        color: white !important;
        padding: 0.8rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-radius: 5px !important;
        border: none !important;
        width: 100% !important;
    }
    
    /* Login link */
    .login-link {
        text-align: center;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .login-link a {
        color: #10B981 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    
    /* Make container narrower */
    .registration-container {
        max-width: 400px !important;
        margin: 0 auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create the centered container for registration
    with st.container():
        st.markdown('<div class="registration-container">', unsafe_allow_html=True)
        
        # Title
        st.title("Create account")
        
        # Registration form
        with st.form("registration_form", clear_on_submit=False):
            # Username input
            username = st.text_input("Username*", key="reg_username")
            
            # Email input
            email = st.text_input("Email address*", key="reg_email")
            
            # Password input
            password = st.text_input("Password*", type="password", key="reg_password")
            
            # Confirm password
            confirm_password = st.text_input("Confirm Password*", type="password", key="reg_confirm_password")
            
            # Submit button styled as green "Create account" button
            st.markdown('<div class="continue-btn">', unsafe_allow_html=True)
            submit = st.form_submit_button("Create account")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submit:
                if not username or not email or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif not utils.is_valid_email(email):
                    st.error("Please enter a valid email address")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    registration_result = user_management.register_user(username, email, password)
                    if registration_result["success"]:
                        st.success("Registration successful! Logging you in…")
                        # mark registration flow as over
                        st.session_state.registration = False

                        # auto-authenticate
                        st.session_state.authenticated = True
                        st.session_state.username      = username

                            # ─── Create a fresh chat session in MongoDB ───
                        new_id = create_chat(username, [], None)
                        st.session_state.current_chat_id = str(new_id)

                        # ─── Initialize an empty chat history in session_state ───
                        st.session_state.chat_history = []

                        # ─── Reload & sort all chats so newest (our “New Chat”) is first ───
                        st.session_state.available_chats = sorted(
                            get_chats_by_user(username),
                            key=lambda m: m["last_updated"],
                            reverse=True
                        )

                        # ─── Force the sidebar to select the very first entry (index 0) ───
                        st.session_state.selected_chat_idx = 0


                        # (optionally set a cookie if you want "remember me" here)

                        st.rerun()
                    else:
                        st.error(registration_result["message"])

        
        
        # Back button (hidden, triggered by the Sign In link)
        with st.container():
            st.markdown('<div class="hide-button">', unsafe_allow_html=True)
            if st.button("Back to Login", key="back_to_login", type="secondary"):
                st.session_state.registration = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)  # Close registration-container

def display_password_reset():
    """Display and handle the password reset process via email and code verification.

    This function renders a two-step password reset interface using Streamlit. In the
    first step, users enter their email to receive a reset code. In the second step,
    users enter the received code along with a new password to complete the reset.

    Returns:
        None
    """

    # Reuse the same CSS from login page
    st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #333 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        padding: 0.8rem !important;
        font-size: 1rem !important;
        border-radius: 5px !important;
        border: 1px solid #ccc !important;
    }
    
    /* Continue button styling */
    .continue-btn button {
        background-color: #10B981 !important;
        color: white !important;
        padding: 0.8rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-radius: 5px !important;
        border: none !important;
        width: 100% !important;
    }
    
    /* Login link */
    .login-link {
        text-align: center;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .login-link a {
        color: #10B981 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    
    /* Make container narrower */
    .reset-container {
        max-width: 400px !important;
        margin: 0 auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # First step: Request email
    if "reset_email_sent" not in st.session_state:
        st.session_state.reset_email_sent = False
        
    # Create the centered container for password reset
    with st.container():
        st.markdown('<div class="reset-container">', unsafe_allow_html=True)
        
        if not st.session_state.reset_email_sent:
            # Title
            st.title("Reset password")
            
            # Password reset request form
            with st.form("reset_request_form", clear_on_submit=False):
                # Email input
                email = st.text_input("Email address*", key="reset_email")
                
                # Submit button styled as green "Send reset link" button
                st.markdown('<div class="continue-btn">', unsafe_allow_html=True)
                submit = st.form_submit_button("Send reset link")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if submit:
                    if not email:
                        st.error("Please enter your email address")
                    elif not utils.is_valid_email(email):
                        st.error("Please enter a valid email address")
                    else:
                        reset_result = user_management.request_password_reset(email)
                        if reset_result["success"]:
                            st.session_state.reset_email_sent = True
                            st.success("Password reset instructions have been sent to your email")
                            st.rerun()
                        else:
                            st.error(reset_result["message"])
            
            # Login link
            st.markdown('<div class="login-link">Remember your password? <a href="#" onclick="document.querySelector(\'[data-testid=\'stForm\'] button[kind=secondaryFormSubmit]\').click();">Back to Sign In</a></div>', unsafe_allow_html=True)
            
            # Back button (hidden, triggered by the Sign In link)
            with st.container():
                st.markdown('<div class="hide-button">', unsafe_allow_html=True)
                if st.button("Back to Login", key="back_to_login", type="secondary"):
                    st.session_state.reset_password = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
        # Second step: Enter reset code and new password
        else:
            # Title
            st.title("Create new password")
            
            # Reset password form
            with st.form("reset_password_form", clear_on_submit=False):
                # Reset code input
                reset_code = st.text_input("Reset code from email*", key="reset_code")
                
                # New password input
                new_password = st.text_input("New Password*", type="password", key="new_password")
                
                # Confirm password
                confirm_password = st.text_input("Confirm New Password*", type="password", key="confirm_new_password")
                
                # Submit button styled as green "Reset Password" button
                st.markdown('<div class="continue-btn">', unsafe_allow_html=True)
                submit = st.form_submit_button("Reset Password")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if submit:
                    if not reset_code or not new_password or not confirm_password:
                        st.error("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 8:
                        st.error("Password must be at least 8 characters long")
                    else:
                        reset_result = user_management.reset_password(reset_code, new_password)
                        if reset_result["success"]:
                            st.success("Password has been reset successfully. You can now login with your new password.")
                            st.session_state.reset_password = False
                            st.session_state.reset_email_sent = False
                            st.rerun()
                        else:
                            st.error(reset_result["message"])
            
            # Login link
            st.markdown('<div class="login-link">Remember your password? <a href="#" onclick="document.querySelector(\'[data-testid=\'stForm\'] button[kind=secondaryFormSubmit]\').click();">Back to Sign In</a></div>', unsafe_allow_html=True)
            
            # Back button (hidden, triggered by the Sign In link)
            with st.container():
                st.markdown('<div class="hide-button">', unsafe_allow_html=True)
                if st.button("Back to Login", key="back_to_login2", type="secondary"):
                    st.session_state.reset_password = False
                    st.session_state.reset_email_sent = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)  # Close reset-container

def display_main_app():
    """Render the main chatbot application interface after user authentication.

    Initializes required session state variables, runs the Streamlit app via
    AppManager, and provides a logout option that clears session and cookie data.

    Returns:
        None
    """

    # ── Make sure our chatbot state exists ──────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mentioned_tickers" not in st.session_state:
        st.session_state.mentioned_tickers = set()
    app = AppManager()
    app.run()
    
    # Logout button
    if st.button("Logout"):
        # Clear session state
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.cookie_checked = False
        st.session_state.chat_history = []
        # Clear auth cookie if it exists
        if cookie_manager.get("auth_token"):
            cookie_manager.delete("auth_token")
        
        st.rerun()
    

# Run the app
if __name__ == "__main__":
    main()
