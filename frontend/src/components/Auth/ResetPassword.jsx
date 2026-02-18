import React, { useState } from "react";
import axios from "axios";
import { useLocation, useNavigate, Link } from "react-router-dom";
import { Eye, EyeOff } from "lucide-react";
import "./login.css";
import darkAlert from "../utils/sweetalert";

function ResetPassword() {
  const location = useLocation();
  const navigate = useNavigate();
  const initialEmail = location.state?.email || "";

  const [email, setEmail] = useState(initialEmail);
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [cooldown, setCooldown] = useState(false);

  async function handleReset(e) {
    e.preventDefault();
    if (cooldown) return;

    setError("");

    if (newPassword !== confirmPassword) {
      setError("Passwords do not match");

      darkAlert.fire({
        icon: "error",
        title: "Password Mismatch",
        text: "New password and confirm password must match",
      });

      return;
    }

    setLoading(true);

    try {
      await axios.post(
        "http://localhost:5000/api/backend/reset-password",
        { email, newPassword, confirmPassword },
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        },
      );

      // ✅ Success alert
      await darkAlert.fire({
        icon: "success",
        title: "Password Updated",
        text: "You can now log in with your new password",
        timer: 1500,
        showConfirmButton: false,
        iconColor: "#22c55e",
      });

      navigate("/login");
    } catch (err) {
      const message =
        err.response?.data?.message || "Reset failed. Please try again.";

      setError(message);

      darkAlert.fire({
        icon: "error",
        title: "Reset Failed",
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
        <h1 className="logo">DD AI</h1>
        <h2>Reset Password</h2>
        <p className="subtitle">Choose a new password for your account.</p>

        <form className="auth-form" onSubmit={handleReset}>
          <input
            type="email"
            name="email"
            placeholder="Email"
            className="auth-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <div style={{ position: "relative" }}>
            <input
              type={showPassword ? "text" : "password"}
              name="newPassword"
              placeholder="New password"
              className="auth-input"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
            />
            <span
              onClick={() => setShowPassword(!showPassword)}
              style={{
                position: "absolute",
                right: 12,
                top: "37%",
                transform: "translateY(-50%)",
                cursor: "pointer",
              }}
            >
              {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
            </span>
          </div>

          <div style={{ position: "relative" }}>
            <input
              type={showConfirmPassword ? "text" : "password"}
              name="confirmPassword"
              placeholder="Confirm new password"
              className="auth-input"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
            <span
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              style={{
                position: "absolute",
                right: 12,
                top: "37%",
                transform: "translateY(-50%)",
                cursor: "pointer",
              }}
            >
              {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
            </span>
          </div>

          {error && <p className="error-msg">{error}</p>}

          <button
            className="auth-btn"
            type="submit"
            disabled={loading || cooldown}
          >
            {loading
              ? "Resetting..."
              : cooldown
                ? "Please wait..."
                : "Reset Password"}
          </button>
        </form>

        <p className="bottom-text">
          Remembered? <Link to="/login">Sign in</Link>
        </p>
      </div>

      <div className="auth-right">
        <div className="feature-card">
          <div className="feature-content">
            <h1>Securely reset</h1>
            <p>Create a strong password to protect your account.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResetPassword;
