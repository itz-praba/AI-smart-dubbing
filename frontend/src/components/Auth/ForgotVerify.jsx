import React, { useState } from "react";
import axios from "axios";
import { useLocation, useNavigate, Link } from "react-router-dom";
import "./login.css";

function ForgotVerify() {
    const location = useLocation();
    const navigate = useNavigate();
    const initialEmail = location.state?.email || "";

    const [email, setEmail] = useState(initialEmail);
    const [otp, setOtp] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    async function handleVerify(e) {
        e.preventDefault();
        setLoading(true);
        setError("");
        try {
            await axios.post(
                "http://localhost:5000/api/backend/otp-validation",
                { email, otp },
                { headers: { "Content-Type": "application/json" }, withCredentials: true }
            );

            navigate("/forgot/reset", { state: { email } });
        } catch (err) {
            setError(err.response?.data?.message || "Invalid or expired OTP");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="auth-container">
            <div className="auth-left">
                <h1 className="logo">DD AI</h1>
                <h2>Verify OTP</h2>
                <p className="subtitle">Enter the OTP sent to your email.</p>

                <form className="auth-form" onSubmit={handleVerify}>
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
                        type="text"
                        name="otp"
                        placeholder="Enter OTP"
                        className="auth-input"
                        value={otp}
                        onChange={e => setOtp(e.target.value)}
                        required
                    />

                    {error && <p className="error-msg">{error}</p>}

                    <button className="auth-btn" type="submit" disabled={loading}>
                        {loading ? "Verifying..." : "Verify OTP"}
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
