"""
Authentication utilities for Google SSO.
"""

import streamlit as st

# Set to True to enable authentication, False to disable
AUTH_ENABLED = False


def check_authentication():
    """
    Check if user is authenticated via Google SSO.
    Returns True if auth is disabled or user is authenticated.
    """
    if not AUTH_ENABLED:
        return True

    try:
        from streamlit_google_auth import Authenticate

        authenticator = Authenticate(
            secret_credentials_path='google_credentials.json',
            cookie_name='auth_cookie',
            cookie_key=st.secrets.get("COOKIE_KEY", "default_secret_key"),
            redirect_uri=st.secrets.get("REDIRECT_URI", "http://localhost:8501"),
        )

        authenticator.check_authentification()

        if st.session_state.get('connected'):
            return True
        else:
            authenticator.login()
            return False

    except Exception as e:
        st.error(f"Authentication error: {e}")
        st.info("Please configure Google OAuth credentials.")
        return False


def get_user_info():
    """
    Get authenticated user info.
    Returns dict with 'email', 'name', 'picture' or None if not authenticated.
    """
    if not AUTH_ENABLED:
        return None

    if st.session_state.get('connected'):
        return {
            'email': st.session_state.get('user_email'),
            'name': st.session_state.get('user_name'),
            'picture': st.session_state.get('user_picture'),
        }
    return None


def logout():
    """Log out the current user."""
    if AUTH_ENABLED:
        try:
            from streamlit_google_auth import Authenticate
            authenticator = Authenticate(
                secret_credentials_path='google_credentials.json',
                cookie_name='auth_cookie',
                cookie_key=st.secrets.get("COOKIE_KEY", "default_secret_key"),
                redirect_uri=st.secrets.get("REDIRECT_URI", "http://localhost:8501"),
            )
            authenticator.logout()
        except Exception:
            pass
