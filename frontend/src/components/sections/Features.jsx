import { motion } from "framer-motion";
import { Mic, ScanFace, Globe, Zap, AudioWaveform } from "lucide-react";

const features = [
  {
    icon: Mic,
    title: "AI Voice Cloning",
    desc: "Clone any speaker's voice with stunning accuracy and natural intonation.",
  },
  {
    icon: ScanFace,
    title: "Lip-Sync Accuracy",
    desc: "Advanced lip-sync technology ensures dubbed audio matches mouth movements.",
  },
  {
    icon: Globe,
    title: "Multi-Language Support",
    desc: "Dub content into 50+ languages with native-sounding pronunciation.",
  },
  {
    icon: Zap,
    title: "Real-Time Processing",
    desc: "Get your dubbed videos in minutes, not days. Lightning fast AI pipeline.",
  },
  {
    icon: AudioWaveform,
    title: "Studio-Quality Output",
    desc: "Professional-grade audio quality that rivals human voice actors.",
  },
];

export default function Features() {
  return (
    <section
      id="features"
      className="relative py-32 bg-gradient-to-b from-black via-neutral-950 to-black"
    >
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
            Powerful <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">Features</span>
          </h2>
          <p className="text-neutral-400 max-w-xl mx-auto text-lg">
            Everything you need to dub content at scale with AI precision.
          </p>
        </motion.div>

        {/* Cards */}
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="group relative rounded-2xl p-6 bg-white/5 backdrop-blur-xl border border-white/10 hover:border-cyan-400/40 transition-all duration-300 shadow-[0_0_40px_rgba(34,211,238,0.06)]"
            >
              <div className="w-12 h-12 rounded-xl bg-cyan-500/10 flex items-center justify-center mb-5 group-hover:bg-cyan-500/20 transition">
                <f.icon className="w-6 h-6 text-cyan-400" />
              </div>

              <h3 className="text-lg font-semibold text-white mb-2">
                {f.title}
              </h3>

              <p className="text-sm text-neutral-400 leading-relaxed">
                {f.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
