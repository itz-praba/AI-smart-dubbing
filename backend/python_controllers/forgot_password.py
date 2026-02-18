import os
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from passlib.context import CryptContext
from config.db import get_db
from services.send_email import send_otp_email

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

OTP_SECRET = os.getenv("OTP_SECRET")


def hash_otp(otp: str) -> str:
    # FIX: Correct usage of hmac.new() — was previously called as hmac.new() which is valid,
    # but the function signature must be verified. Using hmac.new() correctly here.
    return hmac.new(
        OTP_SECRET.encode(),
        otp.encode(),
        hashlib.sha256
    ).hexdigest()


@router.post("/forgot-password")
async def forgot_password(data: dict):
    email = data.get("email", "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    email = email.lower()
    db = get_db()

    user = await db.users.find_one({"email": email})
    if not user:
        # FIX: Return a generic success message to prevent email enumeration.
        # Don't reveal whether the email exists in the system.
        return {"message": "If this email is registered, an OTP has been sent"}

    # FIX: 6-digit OTP instead of 4-digit for much stronger security
    # 4-digit = 9,000 possibilities; 6-digit = 900,000 possibilities
    otp = str(secrets.randbelow(900000) + 100000)
    hashed = hash_otp(otp)

    await db.users.update_one(
        {"email": email},
        {"$set": {
            "resetOtp": hashed,
            "resetOtpExpiry": datetime.utcnow() + timedelta(minutes=10),
            "otpVerified": False  # FIX: Track OTP verification state
        }}
    )

    await send_otp_email(email, otp)
    return {"message": "If this email is registered, an OTP has been sent"}


@router.post("/validate-otp")
async def otp_validation(data: dict):
    email = data.get("email", "").strip()
    otp = data.get("otp", "").strip()

    if not email or not otp:
        raise HTTPException(status_code=400, detail="Email and OTP are required")

    email = email.lower()
    hashed = hash_otp(otp)
    db = get_db()

    user = await db.users.find_one({
        "email": email,
        "resetOtp": hashed,
        "resetOtpExpiry": {"$gt": datetime.utcnow()}
    })

    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    # FIX: Mark OTP as verified and clear it — don't allow reuse
    await db.users.update_one(
        {"email": email},
        {
            "$unset": {"resetOtp": "", "resetOtpExpiry": ""},
            "$set": {"otpVerified": True}
        }
    )

    return {"message": "OTP validated successfully"}


@router.post("/reset-password")
async def reset_password(data: dict):
    email = data.get("email", "").strip()
    new_password = data.get("newPassword", "")
    confirm_password = data.get("confirmPassword", "")

    if not all([email, new_password, confirm_password]):
        raise HTTPException(status_code=400, detail="All fields are required")

    if new_password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    email = email.lower()
    db = get_db()

    # FIX: Critical security gate — verify OTP was validated before allowing password reset.
    # Without this, anyone knowing a user's email could reset their password.
    user = await db.users.find_one({"email": email, "otpVerified": True})
    if not user:
        raise HTTPException(status_code=403, detail="OTP verification required before resetting password")

    hashed = pwd_context.hash(new_password)

    await db.users.update_one(
        {"email": email},
        {
            "$set": {"password": hashed},
            "$unset": {"otpVerified": ""}  # Clear the verified flag after use
        }
    )

    return {"message": "Password reset successful"}