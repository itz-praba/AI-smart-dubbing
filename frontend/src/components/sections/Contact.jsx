import { useState } from "react";
import axios from "axios";
import darkAlert from "../utils/sweetalert";

export default function Contact() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!form.name || !form.email || !form.subject || !form.message) {
      darkAlert.fire({
        icon: "error",
        title: "Missing Fields",
        text: "Please fill in all fields before submitting.",
      });
      return;
    }

    setLoading(true);

    try {
      await axios.post("http://localhost:8001/contact", form);

      darkAlert.fire({
        icon: "success",
        title: "Message Sent!",
        text: "Thanks for reaching out. We'll get back to you soon.",
        confirmButtonText: "Great!",
        iconColor: "#22c55e",
      });

      setForm({
        name: "",
        email: "",
        subject: "",
        message: "",
      });
    } catch (error) {
      darkAlert.fire({
        icon: "error",
        title: "Failed to Send",
        text: "Something went wrong. Please try again later.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-black to-neutral-900 flex items-center justify-center px-6">
      <div className="w-full max-w-5xl grid md:grid-cols-2 gap-10">

        {/* Left Content */}
        <div className="space-y-6">
          <h1 className="text-4xl font-bold text-white">
            Get in <span className="text-cyan-400">Touch</span>
          </h1>
          <p className="text-gray-400">
            Have questions about DubAI? Need help with dubbing or pricing?
            Our team is here to help you anytime.
          </p>

          <div className="space-y-3 text-gray-300">
            <p>📧 support@dubai.ai</p>
            <p>💼 sales@dubai.ai</p>
            <p>🌍 Available worldwide</p>
          </div>
        </div>

        {/* Contact Form */}
        <div className="bg-white/5 backdrop-blur-md border border-cyan-400/20 rounded-2xl p-8 shadow-lg">
          <form className="space-y-5" onSubmit={handleSubmit}>
            <input
              name="name"
              value={form.name}
              onChange={handleChange}
              placeholder="Your Name"
              className="w-full bg-black/40 border border-gray-700 rounded-lg px-4 py-3 text-white"
            />

            <input
              name="email"
              type="email"
              value={form.email}
              onChange={handleChange}
              placeholder="Email Address"
              className="w-full bg-black/40 border border-gray-700 rounded-lg px-4 py-3 text-white"
            />

            <input
              name="subject"
              value={form.subject}
              onChange={handleChange}
              placeholder="Subject"
              className="w-full bg-black/40 border border-gray-700 rounded-lg px-4 py-3 text-white"
            />

            <textarea
              name="message"
              rows="4"
              value={form.message}
              onChange={handleChange}
              placeholder="Your Message"
              className="w-full bg-black/40 border border-gray-700 rounded-lg px-4 py-3 text-white"
            />

            <button
              disabled={loading}
              className={`w-full font-semibold py-3 rounded-lg transition
                ${
                  loading
                    ? "bg-cyan-400/60 cursor-not-allowed"
                    : "bg-cyan-400 hover:bg-cyan-300 text-black"
                }`}
            >
              {loading ? "Sending..." : "Send Message"}
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}
