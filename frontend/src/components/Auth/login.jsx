import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import "./login.css";

const FEATURES = [
    {
        title: "Best features for high-quality result",
        desc: "VoiceClone in 29 languages with ultra realistic AI voice generation."
    },
    {
        title: "Perfect AI Lip Sync",
        desc: "Match translated speech with natural mouth movement automatically."
    },
    {
        title: "Multi Speaker Support",
        desc: "Dub multiple speakers effortlessly with advanced AI detection."
    }
];

function Login() {

    const navigate = useNavigate();
    const carouselRef = useRef(null);

    const [formData, setFormData] = useState({
        email: "",
        password: ""
    });

    const [loading, setLoading] = useState(false);
    const [errMsg, setErrMsg] = useState("");
    const [index, setIndex] = useState(0);

    useEffect(() => {

        const interval = setInterval(() => {

            const nextIndex = (index + 1) % FEATURES.length;
            setIndex(nextIndex);

            carouselRef.current?.scrollTo({
                left: nextIndex * carouselRef.current.offsetWidth,
                behavior: "smooth"
            });

        }, 4000);

        return () => clearInterval(interval);

    }, [index]);

    function handleChange(e) {

        const { name, value } = e.target;

        setErrMsg("");

        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    }

    async function handleSubmit(e) {

        e.preventDefault();
        setLoading(true);

        try {

            const res = await axios.post(
                "http://localhost:5000/api/backend/login",
                formData,
                {
                    headers: {
                        "Content-Type": "application/json"
                    },
                    withCredentials: true
                }
            );

            localStorage.setItem("token", res.data.token);
            navigate("/");

        } catch (err) {

            const message =
                err.response?.data?.message ||
                "Login failed. Please try again.";

            setErrMsg(message);

        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="auth-container">

            <div className="auth-left">

                <h1 className="logo">DD AI</h1>

                <h2>Welcome Back</h2>

                <p className="subtitle">
                    Join millions translating videos with AI
                </p>

                <form className="auth-form" onSubmit={handleSubmit}>

                    <input
                        type="email"
                        name="email"
                        placeholder="Enter your email"
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

                    <div style={{ textAlign: 'right', marginBottom: 12 }}>
                        <Link to="/forgot" className="forgot-link">Forgot password?</Link>
                    </div>

                    {errMsg && (
                        <p className="error-msg">{errMsg}</p>
                    )}

                    <button
                        className="auth-btn"
                        type="submit"
                        disabled={loading}
                    >
                        {loading ? "Signing in..." : "Sign In"}
                    </button>

                </form>

                <p className="bottom-text">
                    Don't have an account?{" "}
                    <Link to="/signup">Sign up</Link>
                </p>

            </div>
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

export default Login;
