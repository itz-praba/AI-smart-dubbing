import React, { useState } from "react";
import axios from "axios";
import { useLocation, useNavigate, Link } from "react-router-dom";
import "./login.css";

function ResetPassword() {
    const location = useLocation();
    const navigate = useNavigate();
    const initialEmail = location.state?.email || "";

    const [email, setEmail] = useState(initialEmail);
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    async function handleReset(e) {
        e.preventDefault();
        setLoading(true);
        setError("");

        try {
            await axios.post(
                "http://localhost:5000/api/backend/reset-password",
                { email, newPassword, confirmPassword },
                { headers: { "Content-Type": "application/json" }, withCredentials: true }
            );

            navigate("/login");
        } catch (err) {
            setError(err.response?.data?.message || "Reset failed. Please try again.");
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
                        onChange={e => setEmail(e.target.value)}
                        required
                    />

                    <input
                        type="password"
                        name="newPassword"
                        placeholder="New password"
                        className="auth-input"
                        value={newPassword}
                        onChange={e => setNewPassword(e.target.value)}
                        required
                    />

                    <input
                        type="password"
                        name="confirmPassword"
                        placeholder="Confirm new password"
                        className="auth-input"
                        value={confirmPassword}
                        onChange={e => setConfirmPassword(e.target.value)}
                        required
                    />

                    {error && <p className="error-msg">{error}</p>}

                    <button className="auth-btn" type="submit" disabled={loading}>
                        {loading ? "Resetting..." : "Reset Password"}
                    </button>
                </form>

                <p className="bottom-text">
                    Remembered? {" "}<Link to="/login">Sign in</Link>
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
