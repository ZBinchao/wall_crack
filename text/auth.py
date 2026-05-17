from functools import wraps
from fastapi import Request, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
import bcrypt

from database import get_db
from models import User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def login_user(request: Request, username: str, password: str, db: Session) -> bool:
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return False
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    request.session["role"] = user.role
    request.session["real_name"] = user.real_name
    return True


def logout_user(request: Request):
    request.session.clear()


def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    if user_id is None:
        return None
    return {
        "id": user_id,
        "username": request.session.get("username"),
        "role": request.session.get("role"),
        "real_name": request.session.get("real_name"),
    }


def require_login(func):
    @wraps(func)
    async def wrapper(*args, request: Request, **kwargs):
        user = get_current_user(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=302)
        return await func(*args, request=request, **kwargs)
    return wrapper


def init_admin_user(db: Session):
    admin = db.query(User).filter(User.username == "admin").first()
    if admin is None:
        admin = User(
            username="admin",
            password_hash=hash_password("admin123"),
            role="admin",
            real_name="管理员",
        )
        db.add(admin)
        db.commit()
