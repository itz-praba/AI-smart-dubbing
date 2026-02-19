import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import "./login.css"; // reuse SAME css
import { ArrowLeft } from "lucide-react";
import { Eye, EyeOff } from "lucide-react";
import darkAlert from "../utils/sweetalert";

const FEATURES = [
  {
    title: "Create your AI account",
    desc: "Start dubbing videos in minutes with powerful AI.",
  },
  {
    title: "Reach Global Audience",
    desc: "Translate content into 130+ languages instantly.",
  },
  {
    title: "Studio Quality Voice",
    desc: "Generate ultra realistic voiceovers.",
  },
];

function Signup() {
  const navigate = useNavigate();

  const [index, setIndex] = useState(0);

  const [formData, setFormData] = useState({
    name: "",
    phone_no: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [cooldown, setCooldown] = useState(false);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % FEATURES.length);
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  function handleChange(e) {
    const { name, value } = e.target;

    setError("");

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (cooldown) return;

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");

      darkAlert.fire({
        icon: "error",
        title: "Password Mismatch",
        text: "Password and confirm password must be the same",
      });

      return;
    }

    setLoading(true);
    setError("");

    // ⏳ Loading alert
    darkAlert.fire({
      title: "Creating your account…",
      allowOutsideClick: false,
      didOpen: () => darkAlert.showLoading(),
    });

    try {
      await axios.post("http://localhost:8001/signup", formData, {
        headers: { "Content-Type": "application/json" },
        withCredentials: true,
      });

      // ✅ Success alert
      await darkAlert.fire({
        icon: "success",
        title: "Account Created",
        text: "You can now log in to your account",
        timer: 1500,
        showConfirmButton: false,
        iconColor: "#22c55e",
      });

      navigate("/login");
    } catch (err) {
      const status = err.response?.status;
      const message =
        status === 409
          ? "User already exists. Try logging in."
          : err.response?.data?.message || "Signup failed. Please try again.";

      setError(message);

      darkAlert.fire({
        icon: "error",
        title: "Signup Failed",
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
      {/* LEFT SIDE */}
      <div className="auth-left">
        <div className="mb-4">
          <button
            className="back-button  text-white  cursor-pointer flex items-center p-2"
            onClick={() => navigate("/login")}
          >
            <ArrowLeft /> back
          </button>
        </div>

        <h2>Create Account</h2>

        <p className="subtitle">Join millions translating videos with AI</p>

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

          <div style={{ position: "relative" }}>
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
                top: "50%",
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
              placeholder="Confirm Password"
              className="auth-input"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
            />
            <span
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              style={{
                position: "absolute",
                right: 12,
                top: "50%",
                transform: "translateY(-50%)",
                cursor: "pointer",
              }}
            >
              {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
            </span>
          </div>

          {error && <p className="error-msg">{error}</p>}

          <button className="auth-btn" disabled={loading || cooldown}>
            {loading
              ? "Creating account..."
              : cooldown
                ? "Please wait..."
                : "Sign Up"}
          </button>
        </form>

        <p className="bottom-text">
          Already have an account? <Link to="/login">Login</Link>
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
