"""
Authentication utilities for Google SSO.
"""

import streamlit as st
import json
import tempfile
import os

# Set to True to enable authentication, False to disable
AUTH_ENABLED = True

# Access control mode:
# - "open": Any Google account can log in
# - "whitelist": Only emails in ALLOWED_USERS can log in
ACCESS_MODE = "whitelist"


def get_credentials_path():
    """
    Get path to Google credentials file.
    Creates a temp file from secrets if running on Streamlit Cloud.
    """
    # Check if credentials file exists locally
    if os.path.exists('google_credentials.json'):
        return 'google_credentials.json'

    # Try to create from secrets
    try:
        # Option 1: JSON string in secrets
        if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
            creds = st.secrets["GOOGLE_CREDENTIALS_JSON"]
            if isinstance(creds, str):
                creds_dict = json.loads(creds)
            else:
                creds_dict = dict(creds)

            # Write to temp file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(creds_dict, temp_file)
            temp_file.close()
            return temp_file.name

        # Option 2: Structured TOML in secrets
        if "google_credentials" in st.secrets:
            creds_dict = dict(st.secrets["google_credentials"])

            # Write to temp file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(creds_dict, temp_file)
            temp_file.close()
            return temp_file.name

    except Exception as e:
        st.error(f"Error loading credentials: {e}")

    return None


def get_allowed_users():
    """Get list of allowed user emails from secrets."""
    try:
        return list(st.secrets.get("ALLOWED_USERS", []))
    except Exception:
        return []


def get_admin_users():
    """Get list of admin user emails from secrets."""
    try:
        return list(st.secrets.get("ADMIN_USERS", []))
    except Exception:
        return []


def is_user_allowed(email):
    """Check if email is in the allowed users list."""
    if ACCESS_MODE == "open":
        return True

    allowed = get_allowed_users()
    admins = get_admin_users()

    # Admins are always allowed
    all_allowed = set(allowed) | set(admins)

    return email.lower() in [e.lower() for e in all_allowed]


def is_admin(email=None):
    """Check if the current user (or specified email) is an admin."""
    if not AUTH_ENABLED:
        return True  # Everyone is admin when auth is disabled

    if email is None:
        email = st.session_state.get('user_email', '')

    if not email:
        return False

    admins = get_admin_users()
    return email.lower() in [e.lower() for e in admins]


def check_authentication():
    """
    Check if user is authenticated via Google SSO.
    Returns True if auth is disabled or user is authenticated and allowed.
    """
    if not AUTH_ENABLED:
        return True

    try:
        from streamlit_google_auth import Authenticate

        creds_path = get_credentials_path()
        if not creds_path:
            st.error("Google credentials not configured.")
            st.info("Please add GOOGLE_CREDENTIALS_JSON to your Streamlit secrets.")
            st.info(f"Available secrets keys: {list(st.secrets.keys()) if hasattr(st, 'secrets') else 'None'}")
            return False

        # Debug: show what's being used
        redirect_uri = st.secrets.get("REDIRECT_URI", "http://localhost:8501")
        st.sidebar.caption(f"Auth: credentials loaded")
        st.sidebar.caption(f"Redirect: {redirect_uri}")

        authenticator = Authenticate(
            secret_credentials_path=creds_path,
            cookie_name='trading_analyzer_auth',
            cookie_key=st.secrets.get("COOKIE_KEY", "default_secret_key"),
            redirect_uri=redirect_uri,
        )

        authenticator.check_authentification()

        if st.session_state.get('connected'):
            email = st.session_state.get('user_email', '')

            # Check if user is allowed
            if not is_user_allowed(email):
                st.error(f"Access denied. {email} is not authorized to use this application.")
                if st.button("Logout"):
                    authenticator.logout()
                    st.rerun()
                return False

            return True
        else:
            authenticator.login()
            return False

    except Exception as e:
        st.error(f"Authentication error: {e}")
        st.code(str(e))  # Show full error
        st.info("Please check your Google OAuth configuration.")
        return False


def get_user_info():
    """
    Get authenticated user info.
    Returns dict with 'email', 'name', 'picture', 'is_admin' or None if not authenticated.
    """
    if not AUTH_ENABLED:
        return None

    if st.session_state.get('connected'):
        email = st.session_state.get('user_email', '')
        return {
            'email': email,
            'name': st.session_state.get('user_name'),
            'picture': st.session_state.get('user_picture'),
            'is_admin': is_admin(email),
        }
    return None


def logout():
    """Log out the current user."""
    if AUTH_ENABLED:
        try:
            from streamlit_google_auth import Authenticate

            creds_path = get_credentials_path()
            if creds_path:
                authenticator = Authenticate(
                    secret_credentials_path=creds_path,
                    cookie_name='trading_analyzer_auth',
                    cookie_key=st.secrets.get("COOKIE_KEY", "default_secret_key"),
                    redirect_uri=st.secrets.get("REDIRECT_URI", "http://localhost:8501"),
                )
                authenticator.logout()
        except Exception:
            pass


def render_user_info_sidebar():
    """Render user info and logout button in sidebar."""
    if not AUTH_ENABLED:
        return

    user = get_user_info()
    if user:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Logged in as:**")
        st.sidebar.markdown(f"{user['name']}")
        st.sidebar.caption(user['email'])

        if user['is_admin']:
            st.sidebar.caption("(Admin)")

        if st.sidebar.button("Logout", key="logout_btn"):
            logout()
            st.rerun()
