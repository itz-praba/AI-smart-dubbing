import os
import re
import smtplib
from email.message import EmailMessage
from fastapi import APIRouter, HTTPException

router = APIRouter()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
# FIX: Recipient email moved to environment variable instead of being hardcoded
CONTACT_RECIPIENT = os.getenv("CONTACT_RECIPIENT_EMAIL")


def is_valid_email(email: str) -> bool:
    # More robust email validation pattern
    pattern = r"^[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+$"
    return re.match(pattern, email) is not None


@router.post("/contact")
async def send_contact_message(data: dict):
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    subject = data.get("subject", "").strip()
    message = data.get("message", "").strip()

    if not all([name, email, subject, message]):
        raise HTTPException(status_code=400, detail="All fields are required")

    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email address")

    if not CONTACT_RECIPIENT:
        raise HTTPException(status_code=500, detail="Server misconfiguration: recipient not set")

    msg = EmailMessage()
    msg["From"] = f"DubAI Contact <{EMAIL_USER}>"
    msg["To"] = CONTACT_RECIPIENT
    msg["Subject"] = f"DubAI Contact - {subject}"
    msg["Reply-To"] = email  # FIX: Added Reply-To so replies go directly to the sender

    msg.set_content(f"""
New contact message from DubAI:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=500, detail="Email authentication failed")
    except smtplib.SMTPException as e:
        print(f"SMTP error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

    return {"message": "Message sent successfully"}