"""
Authentication utilities for Google SSO.
"""

import streamlit as st

# Set to True to enable authentication, False to disable
AUTH_ENABLED = False

# Access control mode:
# - "open": Any Google account can log in
# - "whitelist": Only emails in ALLOWED_USERS can log in
ACCESS_MODE = "whitelist"


def get_allowed_users():
    """Get list of allowed user emails from secrets."""
    try:
        return st.secrets.get("ALLOWED_USERS", [])
    except Exception:
        return []


def get_admin_users():
    """Get list of admin user emails from secrets."""
    try:
        return st.secrets.get("ADMIN_USERS", [])
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

        authenticator = Authenticate(
            secret_credentials_path='google_credentials.json',
            cookie_name='auth_cookie',
            cookie_key=st.secrets.get("COOKIE_KEY", "default_secret_key"),
            redirect_uri=st.secrets.get("REDIRECT_URI", "http://localhost:8501"),
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
        st.info("Please configure Google OAuth credentials.")
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
            authenticator = Authenticate(
                secret_credentials_path='google_credentials.json',
                cookie_name='auth_cookie',
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
