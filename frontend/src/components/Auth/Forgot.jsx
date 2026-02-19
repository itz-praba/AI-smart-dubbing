import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import "./login.css";
import darkAlert from "../utils/sweetalert";

function Forgot() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [cooldown, setCooldown] = useState(false);

  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    if (cooldown) return;

    setLoading(true);
    setMessage("");

    try {
      await axios.post(
        "http://localhost:8001/forgot-password",
        { email },
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        },
      );

      // ✅ Success alert
      await darkAlert.fire({
        icon: "success",
        title: "OTP Sent",
        text: "Check your email for the verification code",
        timer: 1500,
        showConfirmButton: false,
        iconColor: "#22c55e",
      });

      navigate("/forgot/verify", { state: { email } });
    } catch (err) {
      const message =
        err.response?.data?.message || "Request failed. Please try again.";

      // ❌ Error alert
      darkAlert.fire({
        icon: "error",
        title: "Request Failed",
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
        <div className="mb-4">
          <button
            className="back-button  text-white  cursor-pointer flex items-center p-2"
            onClick={() => navigate("/login")}
          >
            <ArrowLeft /> back
          </button>
        </div>
        <h2>Reset Password</h2>

        <p className="subtitle">
          Enter your account email to receive reset instructions.
        </p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <input
            type="email"
            name="email"
            placeholder="Email"
            className="auth-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          {message && <p className="error-msg">{message}</p>}

          <button
            className="auth-btn"
            type="submit"
            disabled={loading || cooldown}
          >
            {loading ? "Sending..." : cooldown ? "Please wait..." : "Send OTP"}
          </button>
        </form>
      </div>

      <div className="auth-right">
        <div className="feature-card">
          <div className="feature-content">
            <h1>Reset securely</h1>
            <p>We will email you instructions to reset your password.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Forgot;
