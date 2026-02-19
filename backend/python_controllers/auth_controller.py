from fastapi import APIRouter, Request, HTTPException
from passlib.context import CryptContext
from datetime import datetime
from config.db import get_db

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/signup")
async def signup(data: dict):
    print("Incoming data:", data)
    name = data.get("name", "").strip()
    phone_no = data.get("phone_no", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    confirmPassword = data.get("confirmPassword", "")

    if not all([name, phone_no, email, password, confirmPassword]):
        raise HTTPException(status_code=400, detail="All fields are required")

    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if password != confirmPassword :
        raise HTTPException(status_code=400, detail="Password do not match")


    email = email.lower()
    db = get_db()

    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = pwd_context.hash(password)

    await db.users.insert_one({
        "name": name,
        "phone_no": phone_no,
        "email": email,
        "password": hashed_password,
        "createdAt": datetime.utcnow()
    })

    return {"message": "User registered successfully"}


@router.post("/login")
async def login(request: Request, data: dict):
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    email = email.lower()
    db = get_db()

    user = await db.users.find_one({"email": email})

    # FIX: Use the same error message for both "not found" and "wrong password"
    # to prevent user enumeration attacks.
    if not user or not pwd_context.verify(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    request.session["user"] = {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "phone_no": user["phone_no"]
    }

    return {"message": "Login successful", "user": request.session["user"]}


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"message": "Logged out"}


@router.get("/me")
async def me(request: Request):
    if "user" not in request.session:
        return {"authenticated": False}

    return {"authenticated": True, "user": request.session["user"]}