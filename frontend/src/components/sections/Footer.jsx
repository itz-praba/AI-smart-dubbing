import {
  Mic,
  Github,
  Linkedin,
  Youtube,
  Instagram,
  ArrowRight,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function Footer() {
  const navigate = useNavigate();

  return (
    <footer className="relative bg-black border-t border-white/10 overflow-hidden">
      {/* Glow background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-[500px] h-[500px] bg-cyan-500/10 blur-3xl rounded-full" />
      </div>

      <div className="relative max-w-7xl mx-auto px-6 py-20">
        {/* Main Footer Grid */}
        <div className="grid gap-12 md:grid-cols-3 items-start">
          {/* Left: Brand */}
          <div className="text-left">
            <div className="flex items-center gap-2 text-white font-semibold mb-4">
              <Mic className="w-5 h-5 text-cyan-400" />
              <span className="text-lg">DubAI</span>
            </div>
            <p className="text-sm text-neutral-400 leading-relaxed max-w-sm">
              DubAI is an AI-powered video dubbing platform that helps creators,
              businesses, and educators reach a global audience instantly.
            </p>
          </div>

          {/* Center: Quick Links */}
          <div className="text-center">
            <h4 className="text-white font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-3 text-sm text-neutral-400">
              {[
                { label: "About", href: "/" },
                { label: "Features", href: "/features" },
                { label: "Pricing", href: "/pricing" },
                { label: "Contact", href: "/contact" },
              ].map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    className="hover:text-cyan-400 transition"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Right: Connect */}
          <div className="text-right">
            <h4 className="text-white font-semibold mb-4">Connect</h4>
            <div className="flex justify-end gap-4 text-neutral-400">
              <a
                href="https://www.instagram.com/itz._praba/"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg border border-white/10 hover:text-cyan-400 hover:border-cyan-400/40 transition"
              >
                <Instagram className="w-4 h-4" />
              </a>

              <a
                href="#"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg border border-white/10 hover:text-cyan-400 hover:border-cyan-400/40 transition"
              >
                <Github className="w-4 h-4" />
              </a>

              <a
                href="#"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg border border-white/10 hover:text-cyan-400 hover:border-cyan-400/40 transition"
              >
                <Linkedin className="w-4 h-4" />
              </a>

              <a
                href="#"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg border border-white/10 hover:text-cyan-400 hover:border-cyan-400/40 transition"
              >
                <Youtube className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
        {/* Bottom Bar */}
        <div className="mt-16 pt-6 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-neutral-500">
            © 2026 DubAI. All rights reserved.
          </p>
          <p className="text-xs text-neutral-500">Built with ❤️ using AI</p>
        </div>
      </div>
    </footer>
  );
}
