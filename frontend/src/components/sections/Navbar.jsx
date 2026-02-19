import { useEffect, useState } from "react";
import { Menu, X, Mic } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const navLinks = [
  { label: "Features", href: "features" },
  { label: "How It Works", href: "how-it-works" },
  { label: "Languages", href: "languages" },
  { label: "Pricing", href: "pricing" },
  { label: "Contact", href: "contact" },
];

export default function Navbar() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  // 🔐 CHECK SESSION ON LOAD
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const res = await axios.get(
          "http://localhost:8001/me",
          { withCredentials: true }
        );
        setAuthenticated(res.data.authenticated === true);
      } catch {
        setAuthenticated(false);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleLogout = async () => {
    await axios.post(
      "http://localhost:8001/logout",
      {},
      { withCredentials: true }
    );
    setAuthenticated(false);
    navigate("/");
  };

  const handleGetStarted = () => {
    navigate("/login");
  };

  // ⏳ Avoid flicker
  if (loading) return null;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-black/60 border-b border-white/10">
      <div className="max-w-9xl px-6 h-16 flex items-center justify-between">
        
        {/* Logo */}
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2 text-white font-semibold"
        >
          <Mic className="w-5 h-5 text-cyan-400" />
          <span>DubAI</span>
        </button>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="text-sm text-neutral-400 hover:text-white transition"
            >
              {link.label}
            </a>
          ))}

          {/* AUTH BUTTON */}
          {!authenticated ? (
            <button
              onClick={handleGetStarted}
              className="ml-2 px-5 py-2 text-sm font-semibold rounded-full bg-cyan-400 text-black hover:opacity-90 transition"
            >
              Get Started
            </button>
          ) : (
            <button
              onClick={handleLogout}
              className="ml-2 px-5 py-2 text-sm font-semibold rounded-full bg-cyan-400 text-black hover:opacity-90 transition"
            >
              Logout
            </button>
          )}
        </div>

        {/* Mobile Toggle */}
        <button
          onClick={() => setOpen(!open)}
          className="md:hidden text-white"
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-black/80 backdrop-blur-xl border-t border-white/10"
          >
            <div className="px-6 py-6 flex flex-col gap-4">
              {navLinks.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="text-sm text-neutral-400 hover:text-white transition"
                >
                  {link.label}
                </a>
              ))}

              {!authenticated ? (
                <button
                  onClick={() => {
                    setOpen(false);
                    handleGetStarted();
                  }}
                  className="mt-2 px-5 py-2 text-sm font-semibold rounded-full bg-cyan-400 text-black"
                >
                  Get Started
                </button>
              ) : (
                <button
                  onClick={() => {
                    setOpen(false);
                    handleLogout();
                  }}
                  className="text-sm text-neutral-400 hover:text-white transition"
                >
                  Logout
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
