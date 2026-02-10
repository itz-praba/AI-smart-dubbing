import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import "./login.css"; // reuse SAME css

const FEATURES = [
    {
        title: "Create your AI account",
        desc: "Start dubbing videos in minutes with powerful AI."
    },
    {
        title: "Reach Global Audience",
        desc: "Translate content into 130+ languages instantly."
    },
    {
        title: "Studio Quality Voice",
        desc: "Generate ultra realistic voiceovers."
    }
];

function Signup() {

    const navigate = useNavigate();

    const [index, setIndex] = useState(0);

    const [formData, setFormData] = useState({
        name: "",
        phone_no: "",
        email: "",
        password: "",
        confirmPassword: ""
    });

    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    React.useEffect(() => {

        const interval = setInterval(() => {
            setIndex(prev => (prev + 1) % FEATURES.length);
        }, 4000);

        return () => clearInterval(interval);

    }, []);

    function handleChange(e) {

        const { name, value } = e.target;

        setError("");

        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    }

    async function handleSubmit(e) {

        e.preventDefault();

        if (formData.password !== formData.confirmPassword) {
            return setError("Passwords do not match");
        }

        const { confirmPassword, ...userData } = formData;

        setLoading(true);

        try {

            await axios.post(
                "http://localhost:5000/api/backend/signup",
                userData,
                {
                    headers: {
                        "Content-Type": "application/json"
                    },
                    withCredentials: true
                }
            );

            navigate("/login");

        } catch (err) {

            const status = err.response?.status;

            if (status === 409) {
                setError("User already exists. Try logging in.");
            }
            else {
                setError(
                    err.response?.data?.message ||
                    "Signup failed. Please try again."
                );
            }

        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="auth-container">

            {/* LEFT SIDE */}
            <div className="auth-left">

                <h1 className="logo">DD AI</h1>
                <h2>Create Account</h2>

                <p className="subtitle">
                    Join millions translating videos with AI
                </p>

                <form className="auth-form" onSubmit={handleSubmit}>

                    <input
                        type="text"
                        name="name"
                        placeholder="Full Name"
                        className="auth-input"
                        value={formData.name}
                        onChange={handleChange}
                        required
                    />

                    <input
                        type="tel"
                        name="phone_no"
                        placeholder="Phone Number"
                        className="auth-input"
                        value={formData.phone_no}
                        onChange={handleChange}
                        required
                    />

                    <input
                        type="email"
                        name="email"
                        placeholder="Email"
                        className="auth-input"
                        value={formData.email}
                        onChange={handleChange}
                        required
                    />

                    <input
                        type="password"
                        name="password"
                        placeholder="Password"
                        className="auth-input"
                        value={formData.password}
                        onChange={handleChange}
                        required
                    />

                    <input
                        type="password"
                        name="confirmPassword"
                        placeholder="Confirm Password"
                        className="auth-input"
                        value={formData.confirmPassword}
                        onChange={handleChange}
                        required
                    />

                    {error && (
                        <p className="error-msg">{error}</p>
                    )}

                    <button
                        className="auth-btn"
                        disabled={loading}
                    >
                        {loading ? "Creating account..." : "Sign Up"}
                    </button>

                </form>

                <p className="bottom-text">
                    Already have an account?{" "}
                    <Link to="/login">Login</Link>
                </p>

            </div>


            {/* RIGHT SIDE (Same Hero Card) */}

            <div className="auth-right">

                <div className="feature-card">

                    <div key={index} className="feature-content">
                        <h1>{FEATURES[index].title}</h1>
                        <p>{FEATURES[index].desc}</p>
                    </div>

                    <div className="dots">
                        {FEATURES.map((_, i) => (
                            <span
                                key={i}
                                onClick={() => setIndex(i)}
                                className={i === index ? "dot active" : "dot"}
                            />
                        ))}
                    </div>

                </div>

            </div>

        </div>
    );
}

export default Signup;
