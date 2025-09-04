# Stock Market AI Chatbot - Login System

A secure Streamlit login page with username/password and social login options for a stock market AI chatbot application.

## Features

- User authentication with username and password
- Social login with Google and Facebook
- Password reset functionality
- Secure password hashing with bcrypt
- JWT token-based authentication
- PostgreSQL database for data storage

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r dependencies.txt
   ```
3. Set up your PostgreSQL database and update the DATABASE_URL in database.py or set it as an environment variable

## Configuration

For Google and Facebook social login, set the following environment variables:
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_SECRET
- FACEBOOK_APP_ID
- FACEBOOK_APP_SECRET
- REDIRECT_URI (your application URL)

## Running the Application

```
streamlit run app.py
```

## Files Overview

- `app.py`: Main Streamlit application with UI components
- `auth_manager.py`: Authentication logic (JWT tokens, password hashing)
- `database.py`: Database connections and functions using SQLAlchemy/PostgreSQL
- `social_auth.py`: Social login functionality
- `user_management.py`: User registration and account management
- `utils.py`: Helper functions