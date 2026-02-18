import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")


def send_otp_email(*, to: str, otp: str):
    """
    Send OTP email (HTML + text)
    Equivalent to nodemailer.sendMail()
    """

    if not EMAIL_USER or not EMAIL_PASS:
        raise RuntimeError("EMAIL_USER or EMAIL_PASS not set")

    msg = EmailMessage()

    # Headers (same as nodemailer)
    msg["From"] = f"AI-SMART-DUBBING <{EMAIL_USER}>"
    msg["To"] = to
    msg["Subject"] = "Your Password Reset OTP"

    # ---------- Plain text ----------
    text_content = f"""
Password Reset Request

Your One-Time Password (OTP) is: {otp}

This OTP is valid for 10 minutes.
If you did not request a password reset, please ignore this email.

AI-SMART-DUBBING — Security Team
"""

    msg.set_content(text_content)

    # ---------- HTML ----------
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Password Reset OTP</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background-color: #f4f6f8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .container {{
      max-width: 520px;
      margin: 40px auto;
      background: #ffffff;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }}
    .header {{
      background: linear-gradient(135deg, #4f46e5, #6366f1);
      color: #ffffff;
      padding: 24px;
      text-align: center;
    }}
    .header h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 600;
    }}
    .content {{
      padding: 32px;
      color: #374151;
      line-height: 1.6;
      font-size: 15px;
    }}
    .otp-box {{
      margin: 24px 0;
      padding: 18px;
      text-align: center;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 6px;
      background: #f9fafb;
      border: 1px dashed #c7d2fe;
      border-radius: 6px;
      color: #111827;
    }}
    .info {{
      font-size: 14px;
      color: #6b7280;
      margin-top: 16px;
    }}
    .footer {{
      padding: 20px;
      text-align: center;
      font-size: 12px;
      color: #9ca3af;
      background: #f9fafb;
    }}
    @media (prefers-color-scheme: dark) {{
      body {{ background-color: #0f172a; }}
      .container {{ background: #020617; }}
      .content {{ color: #e5e7eb; }}
      .otp-box {{
        background: #020617;
        color: #ffffff;
        border-color: #6366f1;
      }}
      .footer {{
        background: #020617;
        color: #94a3b8;
      }}
    }}
  </style>
</head>

<body>
  <div class="container">
    <div class="header">
      <h1>Password Reset Verification</h1>
    </div>

    <div class="content">
      <p>Hello,</p>
      <p>
        We received a request to reset your password.  
        Please use the One-Time Password (OTP) below to continue:
      </p>

      <div class="otp-box">{otp}</div>

      <p class="info">
        ⏳ This OTP is valid for <strong>10 minutes</strong> only.
      </p>

      <p class="info">
        🔒 For your security, never share this OTP with anyone.
      </p>

      <p class="info">
        If you did not request a password reset, you can safely ignore this email.
      </p>

      <p>AI-SMART-DUBBING — Security Team</p>
    </div>

    <div class="footer">
      © {datetime.utcnow().year} AI-SMART-DUBBING. All rights reserved.
    </div>
  </div>
</body>
</html>
"""

    msg.add_alternative(html_content, subtype="html")

    # ---------- Send email ----------
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
    except Exception as e:
        print("OTP email send failed:", e)
        raise
