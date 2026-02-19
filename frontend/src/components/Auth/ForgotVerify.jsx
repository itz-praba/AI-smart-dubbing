import React, { useState } from "react";
import axios from "axios";
import { useLocation, useNavigate, Link } from "react-router-dom";
import "./login.css";
import darkAlert from "../utils/sweetalert";

function ForgotVerify() {
  const location = useLocation();
  const navigate = useNavigate();
  const initialEmail = location.state?.email || "";

  const [email, setEmail] = useState(initialEmail);
  const [otp, setOtp] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [cooldown, setCooldown] = useState(false);

  async function handleVerify(e) {
    e.preventDefault();
    if (cooldown) return;

    setLoading(true);
    setError("");

    try {
      await axios.post(
        "http://localhost:8001/validate-otp",
        { email, otp },
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        },
      );

      // ✅ Success alert
      await darkAlert.fire({
        icon: "success",
        title: "OTP Verified",
        text: "You can now reset your password",
        timer: 1500,
        showConfirmButton: false,
        iconColor: "#22c55e",
      });

      navigate("/forgot/reset", { state: { email } });
    } catch (err) {
      const message = err.response?.data?.message || "Invalid or expired OTP";

      setError(message);

      // ❌ Error alert
      darkAlert.fire({
        icon: "error",
        title: "Verification Failed",
        text: message,
      });

      // ⛔ Cooldown (3s)
      setCooldown(true);
      setTimeout(() => setCooldown(false), 3000);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-left">
        <h2>Verify OTP</h2>
        <p className="subtitle">Enter the OTP sent to your email.</p>

        <form className="auth-form" onSubmit={handleVerify}>
          <input
            type="email"
            name="email"
            placeholder="Email"
            className="auth-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <input
            type="text"
            name="otp"
            placeholder="Enter OTP"
            className="auth-input"
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            required
          />

          {error && <p className="error-msg">{error}</p>}

          <button
            className="auth-btn"
            type="submit"
            disabled={loading || cooldown}
          >
            {loading
              ? "Verifying..."
              : cooldown
                ? "Please wait..."
                : "Verify OTP"}
          </button>
        </form>

        <p className="bottom-text">
          Didn’t receive OTP? <Link to="/forgot">Resend</Link>
        </p>
      </div>

      <div className="auth-right">
        <div className="feature-card">
          <div className="feature-content">
            <h1>Secure verification</h1>
            <p>Enter the one-time code we emailed to you.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ForgotVerify;
