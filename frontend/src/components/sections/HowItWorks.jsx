import { motion } from "framer-motion";
import {
  Upload,
  Languages,
  Cpu,
  Download,
  Sparkles,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const steps = [
  {
    icon: Upload,
    title: "Upload Video",
    desc: "Drag and drop your video in any format. We support MP4, MOV, AVI and more.",
  },
  {
    icon: Languages,
    title: "Select Language & Voice",
    desc: "Choose from 50+ languages and natural AI voices with accent control.",
  },
  {
    icon: Cpu,
    title: "AI Processes Audio",
    desc: "Our AI translates, clones voices, syncs lips, and enhances audio quality.",
  },
  {
    icon: Download,
    title: "Download Dubbed Video",
    desc: "Export your studio-quality dubbed video instantly in high resolution.",
  },
];

export default function HowItWorks() {
  const navigate = useNavigate();

  return (
    <section
      id="how-it-works"
      className="relative py-36 bg-gradient-to-b from-black via-neutral-950 to-black"
    >
      <div className="max-w-6xl mx-auto px-6">

        {/* ================= HEADER ================= */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-28"
        >
          <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
            How It{" "}
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Works
            </span>
          </h2>
          <p className="text-neutral-400 text-lg max-w-2xl mx-auto">
            From upload to export, our AI-powered workflow delivers
            studio-quality dubbing in minutes.
          </p>
        </motion.div>

        {/* ================= STATS ================= */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="grid grid-cols-2 sm:grid-cols-4 gap-8 mb-28 text-center"
        >
          {[
            { label: "Languages Supported", value: "50+" },
            { label: "AI Voices", value: "120+" },
            { label: "Processing Speed", value: "10× Faster" },
            { label: "Accuracy Rate", value: "99%" },
          ].map((stat) => (
            <div key={stat.label}>
              <p className="text-3xl font-bold text-white">{stat.value}</p>
              <p className="text-sm text-neutral-400 mt-1">{stat.label}</p>
            </div>
          ))}
        </motion.div>

        {/* ================= STEPS ================= */}
        <div className="relative grid gap-20 sm:grid-cols-2 lg:grid-cols-4 mb-36">
          <div className="hidden lg:block absolute top-8 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent" />

          {steps.map((step, i) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 }}
              className="relative text-center"
            >
              <div className="relative mx-auto mb-6 w-16 h-16 rounded-2xl bg-cyan-500/10 backdrop-blur border border-cyan-400/20 shadow-[0_0_40px_rgba(34,211,238,0.15)] flex items-center justify-center">
                <step.icon className="w-7 h-7 text-cyan-400" />
                <span className="absolute -top-3 -right-3 w-7 h-7 rounded-full bg-cyan-400 text-black text-xs font-bold flex items-center justify-center">
                  {i + 1}
                </span>
              </div>

              <h3 className="text-white font-semibold mb-2">
                {step.title}
              </h3>
              <p className="text-sm text-neutral-400 leading-relaxed max-w-xs mx-auto">
                {step.desc}
              </p>
            </motion.div>
          ))}
        </div>

        {/* ================= AI EXPLANATION ================= */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto text-center mb-36"
        >
          <Sparkles className="w-10 h-10 mx-auto mb-6 text-cyan-400" />
          <h3 className="text-3xl font-bold text-white mb-4">
            Powered by Advanced AI
          </h3>
          <p className="text-neutral-400 leading-relaxed text-lg">
            DubAI uses state-of-the-art speech synthesis, neural translation,
            and lip-sync models trained on millions of samples to deliver
            natural-sounding, emotionally accurate voiceovers — without
            human intervention.
          </p>
        </motion.div>
        {/* ================= FAQ ================= */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto mb-36"
        >
          <h3 className="text-3xl font-bold text-white text-center mb-16">
            Frequently Asked Questions
          </h3>

          <div className="space-y-6">
            {[
              ["How long does dubbing take?", "Most videos are processed within minutes."],
              ["Is my content secure?", "Yes. All videos are encrypted and private."],
              ["Do voices sound natural?", "Yes. AI voices preserve emotion and tone."],
              ["Can I export HD videos?", "Yes. Full HD and studio-quality audio supported."],
            ].map(([q, a]) => (
              <div
                key={q}
                className="p-6 rounded-xl border border-white/10 bg-black"
              >
                <h4 className="text-white font-semibold mb-2">{q}</h4>
                <p className="text-sm text-neutral-400">{a}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ================= FINAL CTA ================= */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <h3 className="text-4xl font-bold text-white mb-6">
            Dub Once. Reach the World.
          </h3>
          <p className="text-neutral-400 mb-10 max-w-2xl mx-auto">
            Join creators and companies using DubAI to localize content at scale.
          </p>

          <button
            onClick={() => navigate("/start-dubbing")}
            className="px-10 py-4 rounded-xl bg-cyan-400 text-black font-semibold hover:opacity-90 transition shadow-lg shadow-cyan-400/20"
          >
            Start Dubbing Free
          </button>
        </motion.div>

      </div>
    </section>
  );
}
