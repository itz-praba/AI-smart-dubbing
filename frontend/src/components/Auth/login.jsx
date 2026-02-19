import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import "./login.css";
import { ArrowLeft } from "lucide-react";
import { Eye, EyeOff } from "lucide-react";
import darkAlert from '../utils/sweetalert'

const FEATURES = [
  {
    title: "Best features for high-quality result",
    desc: "VoiceClone in 29 languages with ultra realistic AI voice generation.",
  },
  {
    title: "Perfect AI Lip Sync",
    desc: "Match translated speech with natural mouth movement automatically.",
  },
  {
    title: "Multi Speaker Support",
    desc: "Dub multiple speakers effortlessly with advanced AI detection.",
  },
];

function Login() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);
  const [errMsg, setErrMsg] = useState("");
  const [index, setIndex] = useState(0);
  const [cooldown, setCooldown] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      const nextIndex = (index + 1) % FEATURES.length;
      setIndex(nextIndex);
    }, 4000);

    return () => clearInterval(interval);
  }, [index]);

  function handleChange(e) {
    const { name, value } = e.target;

    setErrMsg("");

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (cooldown) return;

    setLoading(true);
    setErrMsg("");

    try {
      const res = await axios.post(
        "http://localhost:8001/login",
        formData,
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        },
      );

      // ✅ Success
      darkAlert.fire({
      icon: "success",
      title: "Welcome Back",
      text: "Login successful",
      timer: 1500,
      showConfirmButton: false,
      iconColor: "#22c55e",
    });

      navigate("/start-dubbing");
    } catch (err) {

      // ❌ Error SweetAlert
      darkAlert.fire({
        icon: "error",
        title: "Login Failed",
        text: "Invalid email or password",
      });


      // ⛔ Auto-disable button for 3 seconds
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
            onClick={() => navigate("/")}
          >
            <ArrowLeft /> Back
          </button>
        </div>

        <h2>Welcome Back</h2>
        <br></br>

        <p className="subtitle">Join millions translating videos with AI</p>

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

          <div  style={{ position: "relative" }}>
            <input
              type={showPassword ? "text" : "password"}
              name="password"
              placeholder="Password"
              className="auth-input"
              value={formData.password}
              onChange={handleChange}
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

          <div style={{ textAlign: "right", marginBottom: 12 }}>
            <Link to="/forgot" className="forgot-link">
              Forgot password?
            </Link>
          </div>

          {errMsg && <p className="error-msg">{errMsg}</p>}

          <button
            className="auth-btn"
            type="submit"
            disabled={loading || cooldown}
          >
            {loading
              ? "Signing in..."
              : cooldown
                ? "Please wait..."
                : "Sign In"}
          </button>
        </form>

        <p className="bottom-text">
          Don't have an account? <Link to="/signup">Sign up</Link>
        </p>
      </div>
      <div className="auth-right">
        <div className="feature-card">
          <div className="feature-content fade">
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
