import hashlib
import hmac
import time

from fastapi import Cookie, HTTPException, Response, status

COOKIE_NAME = "gutcheck_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30


def _sign(expiry: int, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), str(expiry).encode("utf-8"), hashlib.sha256).hexdigest()


def make_session_cookie(secret: str) -> str:
    expiry = int(time.time()) + SESSION_MAX_AGE
    return f"{expiry}.{_sign(expiry, secret)}"


def is_valid_session(cookie_value: str | None, secret: str) -> bool:
    if not cookie_value or "." not in cookie_value:
        return False
    expiry_text, signature = cookie_value.split(".", 1)
    try:
        expiry = int(expiry_text)
    except ValueError:
        return False
    if expiry < int(time.time()):
        return False
    expected = _sign(expiry, secret)
    return hmac.compare_digest(signature, expected)


def set_session_cookie(response: Response, secret: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        make_session_cookie(secret),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=SESSION_MAX_AGE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def require_session(secret: str, cookie_value: str | None = Cookie(default=None, alias=COOKIE_NAME)) -> None:
    if not is_valid_session(cookie_value, secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

