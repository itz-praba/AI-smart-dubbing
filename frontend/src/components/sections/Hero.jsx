import { Play, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useState } from "react";
import heroBg from "../../assets/hero-bg.jpg";
import { checkSession } from "../utils/auth";

const Hero = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const handleStartDubbing = async () => {
    if (loading) return;

    setLoading(true);

    const isAuthenticated = await checkSession();

    if (isAuthenticated) {
      navigate("/start-dubbing");
    } else {
      navigate("/login", {
        state: { redirectTo: "/start-dubbing" },
      });
    }

    setLoading(false);
  };

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-background">
      
      {/* Background */}
      <div className="absolute inset-0">
        <img
          src={heroBg}
          alt=""
          className="w-full h-full object-cover opacity-30"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/80 to-black" />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-5xl mx-auto px-6 text-center pt-32 pb-24">
        
        <span className="inline-block mb-6 px-4 py-1.5 rounded-full text-xs font-medium uppercase
          bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          AI-Powered Dubbing Platform
        </span>

        <h1 className="font-bold text-4xl sm:text-5xl md:text-6xl lg:text-7xl mb-6">
          <span className="text-white">Smart Dubbing for</span>
          <br />
          <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-indigo-400 bg-clip-text text-transparent">
            Global Content
          </span>
        </h1>

        <p className="max-w-2xl mx-auto text-gray-300 mb-12">
          Instantly dub your videos into 50+ languages with AI voice cloning,
          perfect lip-sync, and studio-quality output.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          
          {/* Start Dubbing Button */}
          <button
            onClick={handleStartDubbing}
            disabled={loading}
            className={`group inline-flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl
              font-semibold text-sm transition-all shadow-lg
              ${
                loading
                  ? "bg-emerald-500/60 cursor-not-allowed"
                  : "bg-emerald-500 hover:bg-emerald-400 shadow-emerald-500/20"
              }
              text-black`}
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                Processing...
              </>
            ) : (
              <>
                Start Dubbing
                <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </>
            )}
          </button>

          {/* Demo Button */}
          <a
            href="/how-it-works"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl
              border border-white/15 text-white text-sm hover:bg-white/5"
          >
            <Play className="w-4 h-4" />
            Watch Demo
          </a>

        </div>
      </div>
    </section>
  );
};

export default Hero;
