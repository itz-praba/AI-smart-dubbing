import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import "./login.css";

function Forgot() {
    const [email, setEmail] = useState("");
    const [message, setMessage] = useState("");
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    async function handleSubmit(e) {
        e.preventDefault();
        setLoading(true);
        setMessage("");

        try {
            await axios.post(
                "http://localhost:5000/api/backend/forgot-password",
                { email },
                { headers: { "Content-Type": "application/json" }, withCredentials: true }
            );

            // go to OTP verification step and pass email along
            navigate("/forgot/verify", { state: { email } });
        } catch (err) {
            setMessage(err.response?.data?.message || "Request failed. Please try again.");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="auth-container">

            <div className="auth-left">

             
                <h1 className="logo">DD AI</h1>
                <h2>Reset Password</h2>

                <p className="subtitle">Enter your account email to receive reset instructions.</p>

                <form className="auth-form" onSubmit={handleSubmit}>

                    <input
                        type="email"
                        name="email"
                        placeholder="Email"
                        className="auth-input"
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        required
                    />

                    {message && <p className="error-msg">{message}</p>}

                    <button className="auth-btn" type="submit" disabled={loading}>
                        {loading ? "Sending..." : "Send OTP"}
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
